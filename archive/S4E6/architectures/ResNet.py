import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, n_hidden, n_skipped_blocks, dropout_rate):
        super().__init__()
        block_list = []
        for _ in range(n_skipped_blocks):
            block_list.append(nn.Linear(n_hidden, n_hidden, bias=False))
            nn.init.kaiming_uniform_(block_list[-1].weight, nonlinearity='linear')
            #nn.init.constant_(block_list[-1].bias, 0)
            block_list.append(nn.BatchNorm1d(n_hidden))
            block_list.append(nn.SELU())
            block_list.append(nn.AlphaDropout(dropout_rate))
        self.blocks = nn.ModuleList(block_list)
        self.bn = nn.BatchNorm1d(n_hidden)

    def forward(self, x):
        residual = x
        out = x
        for block in self.blocks:
            out = block(out)
        out = self.bn(out + residual)
        return out
    
class InputBlock(nn.Module):
    def __init__(self, n_input, n_hidden, dropout_rate):
        super().__init__()
        self.linear = nn.Linear(n_input, n_hidden, bias=False)
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')
        #nn.init.constant_(self.linear.bias, 0)
        self.bn = nn.BatchNorm1d(n_hidden)
        self.selu = nn.SELU()
        self.dropout = nn.AlphaDropout(dropout_rate)

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = self.selu(out)
        out = self.dropout(out)
        return out

class OutputBlock(nn.Module):
    def __init__(self, n_hidden, dropout_rate):
        super().__init__()
        #self.linear1 = nn.Linear(n_hidden, n_hidden, bias=False)
        #self.bn = nn.BatchNorm1d(n_hidden)
        #self.selu = nn.SELU()
        #self.dropout = nn.AlphaDropout(dropout_rate)
        #nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='linear')
        #nn.init.constant_(self.linear1.bias, 0)
        self.linear2 = nn.Linear(n_hidden, 1) # maybe bias=False?
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        #out = self.linear1(x)
        #out = self.bn(out)
        #out = self.selu(out)
        #out = self.dropout(out)
        #out = self.linear2(out)
        out = self.linear2(x)
        out = self.sigmoid(out)
        out = out.flatten()
        return out

class ResNet(nn.Module):
    def __init__(self, model_parameters):
        super().__init__()
        self.n_blocks = model_parameters['n_blocks']
        self.n_input = model_parameters['n_input']
        self.n_hidden = model_parameters['n_hidden']
        self.n_skipped_blocks = model_parameters['n_skipped_blocks']
        self.dropout_rate = model_parameters['dropout_rate']
        self.res_blocks = []
        self.input = InputBlock(self.n_input, self.n_hidden, self.dropout_rate)
        self.res_blocks = nn.ModuleList([ResBlock(self.n_hidden, self.n_skipped_blocks, self.dropout_rate) for _ in range(self.n_blocks)])
        self.output = OutputBlock(self.n_hidden, dropout_rate=self.dropout_rate)

    def forward(self, x):
        out = self.input(x)
        for block in self.res_blocks:
            out = block(out)
        out = self.output(out)
        return out