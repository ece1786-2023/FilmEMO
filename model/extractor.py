import torch.nn as nn
import torch
from transformers import GPT2Config, GPT2Model, GPT2ForSequenceClassification

# abstract Extractor class
class Extractor(nn.Module):
    def __init__(self, pre_train_model_name, train=True):
        super().__init__()
        self.model = None
        self._train_ = train
    
    def forward(self, X):
        # the last vector of the hidden state contains the compress information of the input text
        if self._train_:
            self.train(True)
            return self.model(**X).last_hidden_state[:, -1, :]
        else:
            self.eval()
            return self.model(**X).last_hidden_state[:, -1, :].detach()
    
    @torch.no_grad()
    def get_output_shape(self):
        input_ids = torch.zeros(1,30).long()
        extract_feature = self.forward({'input_ids':input_ids})
        return extract_feature.shape

class GPT2Extractor(Extractor):
    def __init__(self, pre_train_model_name, train=True):
        super().__init__(pre_train_model_name, train=train)
        configuration = GPT2Config()
        self.model = GPT2Model(configuration)
        self.pre_train_model_name = pre_train_model_name
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.config = self.model.config