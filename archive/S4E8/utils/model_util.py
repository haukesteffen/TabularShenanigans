import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularTransformer(nn.Module):
    def __init__(self, num_features, num_bins, d_model, num_layers, num_heads, d_ff, dropout):
        super().__init__()
        self.feature_embeddings = nn.ModuleList([nn.Embedding(num_bins, d_model) for _ in range(num_features)])
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_features, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model * num_features, 1)

    def forward(self, x, y=None):
        embedded_features = [embed(x[:, i]) for i, embed in enumerate(self.feature_embeddings)]
        x = torch.stack(embedded_features, dim=1)
        x += self.positional_encoding
        x = self.transformer_decoder(x, x, tgt_is_causal=False)
        x = x.view(x.size(0), -1)
        output = self.fc_out(x)
        if y is None:
            loss = None
        else:
            loss = F.binary_cross_entropy_with_logits(output.squeeze(), y)
        return output, loss