import torch
import torch.nn as nn
import torch.nn.functional as F

class SNN(nn.Module):
    def __init__(self, in_features, p_dropout, n_layers, n_hidden):
        super().__init__()
        self.activation = nn.SELU() 
        self.dropout = nn.AlphaDropout(p=p_dropout)
        self.layers = [nn.Linear(in_features, n_hidden),
                       self.activation,
                       self.dropout]
        for _ in range(n_layers-2):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
            self.layers.append(self.activation)
            self.layers.append(self.dropout)
        self.layers.append(nn.Linear(n_hidden, 1))
        self.layers.append(nn.Sigmoid())
        self.layers = nn.ModuleList(self.layers)
        self.net = nn.Sequential(*self.layers)
        for param in self.net.parameters():
                # biases zero
                if len(param.shape) == 1:
                    nn.init.constant_(param, 0)
                # others using lecun-normal initialization
                else:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        out = self.net(x)
        return out