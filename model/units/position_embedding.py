import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, block_size, n_embd):
        super().__init__()
        self.pos_emb = nn.Embedding(block_size, n_embd)

    def forward(self, x):
        _, block_size = x.shape
        pos = torch.arange(block_size, device=x.device).unsqueeze(0)
        return self.pos_emb(pos)
