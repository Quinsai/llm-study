import torch

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_path, block_size, tokenizer):
        self.tokenizer = tokenizer
        self.block_size = block_size

        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.data = self.tokenizer.encode(text)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        pad_len_x = self.block_size - len(x)
        if pad_len_x > 0:
            x = x + [0] * pad_len_x
        pad_len_y = self.block_size - len(y)
        if pad_len_y > 0:
            y = y + [-100] * pad_len_y
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)