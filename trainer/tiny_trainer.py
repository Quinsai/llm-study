from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from dataset.pretrain.dataset import CharDataset
from dataset.sft.dataset import SFTDataset
from model.model import TinyModel
from utils.visualizer import draw_line_plot

class TinyTrainer:
    def __init__(self, corpus_path, sft_message_path, block_size, batch_size, num_epoch_pretrain, num_epoch_sft, iter_per_epoch, lr, fixed_prompt1, fixed_prompt2, tokenizer):
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_epoch_pretrain = num_epoch_pretrain
        self.num_epoch_sft = num_epoch_sft
        self.iter_per_epoch= iter_per_epoch
        self.fixed_prompt1 = fixed_prompt1
        self.fixed_prompt2 = fixed_prompt2
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size

        self.pretrain_dataset = CharDataset(corpus_path, block_size=block_size, tokenizer=self.tokenizer)

        self.sft_dataset = SFTDataset(sft_message_path, block_size=block_size, tokenizer=self.tokenizer)

        self.model = TinyModel(self.vocab_size, block_size=self.block_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def pretrain(self, draw=False, save_path=None):
        data_loader = DataLoader(self.pretrain_dataset, batch_size=self.batch_size, shuffle=True)
        epoch_loss_list = []
        for epoch in range(self.num_epoch_pretrain):
            epoch_loss = 0
            for i, (x, y) in enumerate(data_loader):
                self.optimizer.zero_grad()
                logits, loss = self.model(x,y)
                loss.backward()
                epoch_loss += loss.item()
                self.optimizer.step()
                if i >= self.iter_per_epoch:
                    break
            epoch_loss = epoch_loss / self.iter_per_epoch
            epoch_loss_list.append(epoch_loss)
            if (epoch+1) % 50 == 0:
                print(f"Epoch {epoch+1} average loss {epoch_loss:.4f}")
        if draw:
            draw_line_plot(epoch_loss_list, "epoch", "loss", "pretrain")
        if save_path is not None:
            self.save_checkpoint(save_path)

    def sft_train(self, draw=False, save_path=None):
        data_loader = DataLoader(self.sft_dataset, batch_size=self.batch_size, shuffle=True)
        epoch_loss_list = []
        for epoch in range(self.num_epoch_sft):
            epoch_loss = 0
            for i, (x, y) in enumerate(data_loader):
                self.optimizer.zero_grad()
                logits, loss = self.model(x,y)
                loss.backward()
                epoch_loss += loss.item()
                self.optimizer.step()
                if i >= self.iter_per_epoch:
                    break
            epoch_loss = epoch_loss / self.iter_per_epoch
            epoch_loss_list.append(epoch_loss)
            if (epoch+1) % 50 == 0:
                print(f"Epoch {epoch+1} average loss {epoch_loss:.4f}")
        if draw:
            draw_line_plot(epoch_loss_list, "epoch", "loss", "sft")
        if save_path is not None:
            self.save_checkpoint(save_path)

    def generate(self, prompt):
        input_text = self.fixed_prompt1 + prompt + self.fixed_prompt2
        input_idx = self.tokenizer.encode(input_text)
        input_idx = torch.tensor([input_idx], dtype=torch.long)
        output_idx = self.model.generate(input_idx)
        output_idx = output_idx[0].tolist()
        output_text = self.tokenizer.decode(output_idx)
        _, _, output_text = output_text.partition("###Agent:")
        if output_text[0] == '\n':
            output_text = output_text[1:]
        return output_text
    
    def save_checkpoint(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
        }, path)

    def load_checkpoint(self, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        self.model.load_state_dict(ckpt["model_state"])