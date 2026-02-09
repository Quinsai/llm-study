import torch
import numpy as np

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, bin_prefix, block_size):
        self.block_size = block_size
        self.x_list = np.memmap(bin_prefix+"_x.bin", dtype=np.int32, mode="r")
        self.y_list = np.memmap(bin_prefix+"_y.bin", dtype=np.int32, mode="r")
        self.length = len(self.x_list) // block_size

    def __getitem__(self, index):
        start = index * self.block_size
        end = start + self.block_size
        x = self.x_list[start:end]
        y = self.y_list[start:end]
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))
    
    def __len__(self):
        return self.length

def build_sft_bin(message_path, tokenizer, block_size, out_prefix):
    x_list = []
    y_list = []
    with open(message_path, "r", encoding="utf-8") as f:
        message = ""
        while True:
            line = f.readline()
            if line == "#----end----#\n":
                encoded_message = tokenizer.encode(message, add_bos=False, add_eos=False)
                encoded_message = encoded_message[:block_size]
                encoded_answer_start = tokenizer.encode("###Agent:\n", add_bos=False, add_eos=False)
                start_of_answer = 0
                for i in range(len(encoded_message) - len(encoded_answer_start) + 1):
                    if encoded_message[i:i+len(encoded_answer_start)] == encoded_answer_start:
                        start_of_answer = i
                        break
                x = encoded_message[:-1]
                y = encoded_message[1:]
                pad_len_x = block_size - len(x)
                if pad_len_x > 0:
                    x = x + [0] * pad_len_x
                pad_len_y = block_size - len(y)
                if pad_len_y > 0:
                    y = y + [-100] * pad_len_y
                y[:start_of_answer] = [-100] * start_of_answer
                # x = torch.tensor(x, dtype=torch.long)
                # y = torch.tensor(y, dtype=torch.long)
                x_list.append(x)
                y_list.append(y)
                message = ""
                continue
            elif not line:
                break
            message += line
    x_list = np.array(x_list, dtype=np.int32)
    y_list = np.array(y_list, dtype=np.int32)

    x_list.tofile(out_prefix + "_x.bin")
    y_list.tofile(out_prefix + "_y.bin")