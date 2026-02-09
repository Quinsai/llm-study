import torch
import numpy as np

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, bin_path, block_size):
        self.block_size = block_size
        self.data = np.memmap(bin_path, dtype=np.int32, mode="r")
        self.length = len(self.data) - self.block_size - 1

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        # pad_len_x = self.block_size - len(x)
        # if pad_len_x > 0:
        #     x = x + [0] * pad_len_x
        # pad_len_y = self.block_size - len(y)
        # if pad_len_y > 0:
        #     y = y + [-100] * pad_len_y
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
def build_pretrain_bin(corpus_path, tokenizer, out_path):
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    data = tokenizer.encode(text)
    data = np.array(data, dtype=np.int32)
    data.tofile(out_path)