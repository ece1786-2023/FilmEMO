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
        # hidden layer
        for i in range(len(self.hidden_units)):
            if i == 0:
                input_size = self.number_features
            else:
                input_size = self.hidden_units[i-1]
            output_size = self.hidden_units[i]
            model.append(nn.Linear(input_size,output_size))
            model.append(nn.ReLU())
        # output layer
        if self.number_classes == 2:
            model.append(nn.Linear(self.hidden_units[-1], 1))
            model.append(self.sigmod)
        else:
            model.append(nn.Linear(self.hidden_units[-1], self.number_classes))
            model.append(self.softmax)
        return model
    
    def forward(self, X):
        return self.model(X)
    
    # only forward
    @torch.no_grad()
    def inference(self, X):
        self.eval()
        return self.model(X).detach()