
import numpy as np
import torch
from torch import nn
from transformers import LogitsProcessor
from enum import Enum, auto

class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()

model_layer_to_capture = {
    KEY_TYPE.last_ffn_input: {
        True: lambda model: model.base_model.decoder.layers[-1].fc1,
        False: lambda model: model.base_model.h[-1].mlp,
    }, 
    KEY_TYPE.last_ffn_output: {
        True: lambda model: model.base_model.decoder.layers[-1], 
        False: lambda model: model.base_model.h[-1],
    }
}

class KNNWrapper(LogitsProcessor):
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.prompt_input_ids = None
        self.lmbda = args.lmbda
        self.knn_temperature = args.knn_temp
        self.is_encoder_decoder = model.config.is_encoder_decoder

        layer_to_capture = model_layer_to_capture[args.knn_keytype][self.is_encoder_decoder](model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=args.knn_keytype is KEY_TYPE.last_ffn_input)

        self.keys = None
        self.values = None

    def reset_generation(self, input_ids, **kwargs):
        self.prompt_input_ids = input_ids
        self.keys = None
        self.values = None
        
        if self.is_encoder_decoder:
            self.prompt_attention_mask = kwargs['attention_mask']
            __ = self.model.base_model.decoder(
                    input_ids
                    # **kwargs
                )
            
            self.keys = self.activation_capturer.captured
            self.values = self.prompt_input_ids[:, 1:]

    def break_into(self, model):
        # Inject our hook function in the beginning of generation.
        # When the "model.generate()" will be called, it will first call our "reset_generation()" function, 
        # and only then call "model.generate()"
        self.original_generate_func = model.generate
        
        def pre_generate_hook(input_ids, **kwargs):
            self.reset_generation(input_ids, **kwargs)
            return self.original_generate_func(input_ids, **kwargs)

        model.generate = pre_generate_hook

        # Inject a logits processor to process the logits after the prediction
        # When the model.generate() will call "_get_logits_processor", it will add our class as an additional logits processor,
        # and thus will call our __call__() function below
        self.original_logits_processor_func = model._get_logits_processor

        def get_logits_processor_post_hook(**kwargs):
            logits_processor_list = self.original_logits_processor_func(**kwargs)
            logits_processor_list.append(self)
            return logits_processor_list
        
        model._get_logits_processor = get_logits_processor_post_hook

        # Save a reference to the wrapper
        model.knn_wrapper = self
    
    def break_out(self, model):
        model._get_logits_processor = self.original_logits_processor_func

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if not self.is_encoder_decoder and self.keys is None:
            prompt_length = self.prompt_input_ids.shape[1] - 1
            self.keys = self.activation_capturer.captured[:, :prompt_length]
            self.values = self.prompt_input_ids[:, 1:prompt_length+1]

        query = self.activation_capturer.captured[:, -1]

        # if num_beams > 1:
        # expanded_return_idx = (
        #     torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        # )
        # input_ids = input_ids.index_select(0, expanded_return_idx)

        neg_dist = self.dist_func(query, self.keys)
        knn_log_probs = torch.nn.functional.softmax(neg_dist / self.knn_temperature, dim=-1) # (1, keys)
        knn_log_probs = torch.full(scores.shape, 0.0).scatter_add(dim=1, index=self.values, src=knn_log_probs).log() # (1, vocab)
        knn_log_probs[knn_log_probs == float('-inf')] = -10000.0

        if self.model.config.num_beams == 1:
            scores = torch.nn.functional.log_softmax(scores, dim=-1)
        else:
            # Uri: scores were already softmaxed in beam search
            pass
        interpolated_scores = self.interpolate(knn_log_probs, scores, self.lmbda)
        
        return interpolated_scores

    def dist_func(self, query, keys):
        return -torch.sum((query.unsqueeze(0) - keys)**2, dim=2)

    def interpolate(self, knn_log_probs, lm_log_probs, lmbda):
        combine_probs = torch.stack([lm_log_probs, knn_log_probs], dim=0)
        combine_probs[0] += np.log(1 - lmbda)
        combine_probs[1] += np.log(lmbda)
        curr_prob = torch.logsumexp(combine_probs, dim=0)

        return curr_prob

class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None

        def hook_fn(module, input, output):
            if self.capture_input:
                self.captured = input[0].detach()
            
            else:
                self.captured = output.detach()

        layer.register_forward_hook(hook_fn)

