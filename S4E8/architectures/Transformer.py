import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, head_size, dropout, d_model):
        super().__init__()
        self.key = nn.Linear(d_model, head_size)
        self.query = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        w = k @ q.transpose(-2, -1) * C**-0.5 # (B, T, T), multiply with C**-0.5 to ensure unit gaussian outputs
        w = F.softmax(w, dim=-1) # (B, T, T)
        w = self.dropout(w)
        out = w @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, n_heads, d_model, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, dropout, d_model) for _ in range(n_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.net(x)
        return out
    
class Block(nn.Module):
    def __init__(self, head_size, d_model, n_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(head_size, n_heads, d_model, dropout)
        self.ff = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        attributes = ['d_model', 'n_embed', 'n_heads', 'head_size', 'dropout', 'n_in']
        assert all(hasattr(config, a) for a in attributes)
        self.d_model = config.d_model
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads
        self.head_size = config.head_size
        self.dropout = config.dropout
        self.n_in = config.n_in
        self.embed = nn.Embedding(self.n_in, self.d_model)
        self.blocks = nn.Sequential(
            Block(self.head_size, self.d_model, self.n_heads, self.dropout),
            Block(self.head_size, self.d_model, self.n_heads, self.dropout),
            Block(self.head_size, self.d_model, self.n_heads, self.dropout)
        )
        self.linear = nn.Linear(self.d_model*self.n_in, 1)

    def forward(self, x, y=None):
        out = self.embed(x)
        out = self.blocks(out).view(-1, self.d_model*self.n_in)
        out = self.linear(out).squeeze()

        if y == None:
            loss = None
        else:
            loss = F.binary_cross_entropy_with_logits(out, y)
        return out, loss