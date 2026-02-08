import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()

        self.n_head = n_head
        self.n_embd = n_embd

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        block_size = x.shape[1]
        h = self.ln1(x)

        resi1 = h
        mask = torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).expand((batch_size * self.n_head, block_size, block_size))
        mask = (1-mask) * -1e9
        h, _ = self.attn(h, h, h, attn_mask=mask)
        h = h + resi1
        h = self.ln2(h)

        resi2 = h
        h = self.ff(h)
        h = h + resi2

        return h
