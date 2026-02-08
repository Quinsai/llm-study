import torch
import torch.nn as nn
import torch.nn.functional as F
from .units.linear_head import LinearHead
from .units.position_embedding import PositionalEmbedding
from .units.token_embedding import TokenEmbedding
from .units.transformer_block import TransformerBlock

class TinyModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=256):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.token_embd = TokenEmbedding(vocab_size, n_embd)
        self.position_embd = PositionalEmbedding(block_size, n_embd)
        self.net = nn.Sequential(
            TransformerBlock(n_head=16, n_embd=n_embd),
            TransformerBlock(n_head=16, n_embd=n_embd),
            TransformerBlock(n_head=16, n_embd=n_embd),
            TransformerBlock(n_head=16, n_embd=n_embd),
            TransformerBlock(n_head=16, n_embd=n_embd),
            TransformerBlock(n_head=16, n_embd=n_embd),
            TransformerBlock(n_head=16, n_embd=n_embd),
            TransformerBlock(n_head=16, n_embd=n_embd),
            nn.LayerNorm(n_embd)
        )
        self.output_head = LinearHead(n_embd=n_embd, vocab_size=vocab_size)
        self.output_head.lh.weight = self.token_embd.token_emb.weight

    def forward(self, x, y=None):
        h = self.token_embd(x) + self.position_embd(x)
        h = self.net(h)
        return self.output_head(h, y)
    
    def generate(self, idx, max_new_tokens=10):
        self.eval()
        for _ in range(max_new_tokens):
            idx = idx[:,-self.block_size:]
            with torch.no_grad():
                logits = self.forward(idx)
            next_logits = logits[:,-1,:]
            next_prob = F.softmax(next_logits, dim=-1)
            # 按概率采样
            next_token = torch.multinomial(next_prob, num_samples=1)
            next_token = next_token.clamp(0, self.vocab_size-1)
            # 直接argmax
            # next_token = torch.argmax(next_prob, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
