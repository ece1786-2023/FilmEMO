import torch.nn as nn
import torch
from transformers import GPT2Config, GPT2Model

# abstract Extractor class
class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def forward(self, X, train=True):
        # the last vector of the hidden state contains the compress information of the input text
        if train:
            self.model.train(True)
            return self.model(**X).last_hidden_state[:, -1, :]
        else:
            return self.model(**X).last_hidden_state[:, -1, :].detach()
    
    @torch.no_grad()
    def get_output_shape(self):
        input_ids = torch.zeros(1,30).long()
        extract_feature = self.forward({'input_ids':input_ids})
        return extract_feature.shape

class GPT2Extractor(Extractor):
    def __init__(self):
        super().__init__()
        configuration = GPT2Config()
        self.model = GPT2Model(configuration)
        self.config = self.model.config

# gpt2 = GPT2Extractor()
# print(gpt2.get_output_shape())
