import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, d_model, head_size, dropout):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.dropout = dropout
        self.key = nn.Linear(self.d_model, self.head_size)
        self.query = nn.Linear(self.d_model, self.head_size)
        self.value = nn.Linear(self.d_model, self.head_size)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, self.head_size)
        q = self.query(x) # (B, T, self.head_size)
        v = self.value(x) # (B, T, self.head_size)
        w = k @ q.transpose(-2, -1) * C**-0.5 # (B, T, T), multiply with C**-0.5 to ensure unit gaussian outputs
        w = F.softmax(w, dim=-1) # (B, T, T)
        w = self.dropout(w)
        out = w @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, n_heads, d_model, dropout):
        super().__init__()
        self.head_size = head_size
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.heads = nn.ModuleList([Head(self.d_model, self.head_size, self.dropout) for _ in range(self.n_heads)])
        self.proj = nn.Linear(self.n_heads * self.head_size, self.d_model)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(self.d_model, 4*self.d_model),
            nn.ReLU(),
            nn.Linear(4*self.d_model, self.d_model),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        out = self.net(x)
        return out
    
class Block(nn.Module):
    def __init__(self, head_size, d_model, n_heads, dropout):
        super().__init__()
        self.head_size = head_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.attention = MultiHeadAttention(self.head_size, self.n_heads, self.d_model, self.dropout)
        self.ff = FeedForward(self.d_model, self.dropout)
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert self.config.n_embed % self.config.n_heads == 0
        self.embedcat = nn.Embedding(2, self.config.d_model)
        self.embednum = nn.Embedding(self.config.n_embed, self.config.d_model)
        self.blocks = nn.Sequential(
            Block(self.config.head_size, self.config.d_model, self.config.n_heads, self.config.dropout),
            Block(self.config.head_size, self.config.d_model, self.config.n_heads, self.config.dropout),
            Block(self.config.head_size, self.config.d_model, self.config.n_heads, self.config.dropout),
            Block(self.config.head_size, self.config.d_model, self.config.n_heads, self.config.dropout)
        )
        self.linear = nn.Linear(self.config.d_model * (self.config.n_in_cat + self.config.n_in_num), 3)

    def forward(self, x, y=None):
        xcat = x[:, :self.config.n_in_cat]
        xnum = x[:, self.config.n_in_cat:self.config.n_in_cat + self.config.n_in_num]
        ecat = self.embedcat(xcat)
        enum = self.embednum(xnum)
        out = torch.cat([ecat, enum], dim=1)
        out = self.blocks(out).view(-1, self.config.d_model * (self.config.n_in_cat + self.config.n_in_num))
        out = self.linear(out).squeeze()

        if y == None:
            loss = None
        else:
            loss = F.binary_cross_entropy_with_logits(out, y)
        return out, loss