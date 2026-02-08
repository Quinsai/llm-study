import torch

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, message_path, block_size, tokenizer):
        self.samples = []
        self.block_size = block_size
        self.tokenizer = tokenizer
        with open(message_path, "r", encoding="utf-8") as f:
            message = ""
            while True:
                line = f.readline()
                if line == "\n":
                    message = message[:block_size]
                    start_of_answer = message.find("###Agent:\n") - 1 + 10
                    x = self.tokenizer.encode(message[:-1], add_bos=False, add_eos=False)
                    y = self.tokenizer.encode(message[1:], add_bos=False, add_eos=False)
                    pad_len_x = block_size - len(x)
                    if pad_len_x > 0:
                        x = x + [0] * pad_len_x
                    pad_len_y = block_size - len(y)
                    if pad_len_y > 0:
                        y = y + [-100] * pad_len_y
                    y[:start_of_answer] = [-100] * start_of_answer
                    x = torch.tensor(x, dtype=torch.long)
                    y = torch.tensor(y, dtype=torch.long)
                    self.samples.append((x,y))
                    message = ""
                    continue
                elif not line:
                    break
                message += line

    def __getitem__(self, index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)