import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearHead(nn.Module):
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.lh = nn.Linear(self.n_embd, self.vocab_size, bias=False)
    
    def forward(self, x, y=None):
        y_hat = self.lh(x)
        if y is not None:
            loss = F.cross_entropy(y_hat.view(-1, y_hat.size(-1)), y.view(-1), ignore_index=-100)
            return y_hat, loss
        return y_hat