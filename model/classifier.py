import torch.nn as nn
import torch

#abstract class
class Classifier(nn.Module):
    def __init__(self, number_features, number_classes, hidden_units=[64, 32]):
        super().__init__()
        self.number_features = number_features
        self.number_classes = number_classes
        self.hidden_units = hidden_units
        self.sigmod = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.model = self.get_model()
    
    def get_model(self):
        model = nn.Sequential()
        input_size = self.number_features
        # hidden layer
        for i in range(len(self.hidden_units)):
            output_size = self.hidden_units[i]
            model.append(nn.Linear(input_size,output_size))
            model.append(nn.Dropout(p=0.1))
            model.append(nn.ReLU())
            input_size = output_size
        # output layer
        if self.number_classes == 2:
            model.append(nn.Linear(input_size, 1))
        else:
            model.append(nn.Linear(input_size, self.number_classes))
        return model
    
    def forward(self, X):
        return self.model(X)
    
    # only forward
    @torch.no_grad()
    def inference(self, X):
        self.eval()
        if self.number_classes == 2:
            return self.sigmod(self.model(X).detach())
        else:
            return self.softmax(self.model(X).detach())