
import numpy as np
import torch
from torch import nn
from transformers import LogitsProcessor
from enum import Enum, auto

class KNNWrapper(LogitsProcessor):
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.prompt_input_ids = None
        self.lmbda = args.lmbda
        self.knn_temperature = args.knn_temp

        if args.knn_keytype is KEY_TYPE.last_ffn_input:
            self.keys_capturer = ActivationCapturer(model.base_model.h[-1].mlp, capture_input=True)
        elif args.knn_keytype is KEY_TYPE.last_layer_output:
            self.keys_capturer = ActivationCapturer(model.base_model.h[-1], capture_output=True)

        self.keys = None
        self.values = None

    def reset_generation(self, input_ids):
        self.prompt_input_ids = input_ids
        self.keys = None
        self.values = None

    def break_into(self, model):
        # Inject our function in the beginning of generation
        self.original_generate_func = model.generate
        
        def fake_generate(input_ids, **kwargs):
            self.reset_generation(input_ids)
            return self.original_generate_func(input_ids, **kwargs)

        model.generate = fake_generate

        # Inject a logits processor to process the logits after the prediction
        self.original_logits_processor_func = model._get_logits_processor

        def fake_get_logits_processor(**kwargs):
            logits_processor_list = self.original_logits_processor_func(**kwargs)
            logits_processor_list.append(self)
            return logits_processor_list
        
        model._get_logits_processor = fake_get_logits_processor
    
    def break_out(self, model):
        model._get_logits_processor = self.original_logits_processor_func

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.keys is None:
            prompt_length = self.prompt_input_ids.shape[1] - 1
            self.keys = self.keys_capturer.captured[:, :prompt_length]
            self.values = self.prompt_input_ids[:, 1:prompt_length+1]

        query = self.keys_capturer.captured[:, -1]

        neg_dist = self.dist_func(query, self.keys)
        knn_log_probs = torch.nn.functional.softmax(neg_dist / self.knn_temperature, dim=-1) # (1, keys)
        knn_log_probs = torch.full(scores.shape, 0.0).scatter_add(dim=1, index=self.values, src=knn_log_probs).log() # (1, vocab)

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
    def __init__(self, layer, capture_input=False, capture_output=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input
        self.capture_output = capture_output

        self.captured = None

        def hook_fn(module, input, output):
            if self.capture_input:
                self.captured = input[0].detach()
            
            if self.capture_output:
                self.captured = output.detach()

        layer.register_forward_hook(hook_fn)

        
class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()