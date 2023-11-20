import torch.nn as nn
import torch.nn.functional as Fun
import torch
 
# This model is working for movie reivew sentiment classification
class EmoFilmSystem(nn.Module):
    def __init__(self, tokenizer, feature_extractor, classifier, Lossfn=nn.CrossEntropyLoss, max_length=60):
        super().__init__()
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.Lossfn = Lossfn()
        self.max_length = max_length
    
    def tokenized_input(self, input_text):
        return self.tokenizer(input_text, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
    
    # the X should be a dict which has input_ids, attention_mask
    def forward(self, X):
        extract_feature = self.feature_extractor(X)
        return self.classifier(extract_feature)
    
    def get_loss(self, X, y):
        y_hat = self.forward(X)
        loss = self.Lossfn(y_hat, y)
        return loss, y_hat
    
    def get_number_classes(self):
        return self.classifier.number_classes
    
    def get_pre_train_model_name(self):
        return self.feature_extractor.pre_train_model_name

    def get_deviec(self):
        return list(self.parameters())[0].device

    # predict the label of the input movie review
    @torch.no_grad()
    def inference(self, input_text, need_tokenize=False):
        self.eval()
        # only inference here
        self.feature_extractor._train_ = False
        if need_tokenize:
          tokens = {k:v.to(self.get_deviec()) for k,v in self.tokenized_input(input_text).items()}
        else:
          tokens = input_text
        extract_feature = self.feature_extractor(tokens)
        probabilities = self.classifier.inference(extract_feature)
        return probabilities.argmax(dim=-1)