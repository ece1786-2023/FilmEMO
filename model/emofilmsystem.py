import torch.nn as nn
import torch.nn.functional as Fun
import torch
 


# This model is working for movie reivew sentiment classification
class EmoFilmSystem(nn.Module):
    def __init__(self, tokenizer, feature_extractor, classifier, Lossfn=nn.CrossEntropyLoss):
        super().__init__()
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.Lossfn = Lossfn()
    
    
    def tokenized_input(self, input_text):
        return self.tokenizer(input_text, return_tensors="pt")
    
    # the X should be a dict which has input_ids, attention_mask
    def forward(self, X):
        extract_feature = self.feature_extractor(X)
        return self.classifier(extract_feature)
    
    def get_loss(self, X, y):
        self.train(True)
        y_hat = self.forward(X)
        loss = self.Lossfn(y_hat, y)
        return loss
    
    # predict the label of the input movie review
    @torch.no_grad()
    def inference(self, input_text):
        self.eval()
        tokens = self.tokenized_input(input_text)
        extract_feature = self.feature_extractor(tokens, train=False)
        probabilities = self.classifier.inference(extract_feature)
        return probabilities.argmax(dim=-1)
    
 
