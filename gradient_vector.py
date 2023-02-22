import torch
import torch.nn as nn



class SimpleNN(nn.Module):
    def __init__(self, feature_dim,device):
        super(SimpleNN, self).__init__()
        self.device = device
        self.net = nn.Linear(feature_dim, feature_dim)
        self.feature_dim = feature_dim
    def forward(self):
        return self.net(torch.ones(1,self.feature_dim).to(self.device))