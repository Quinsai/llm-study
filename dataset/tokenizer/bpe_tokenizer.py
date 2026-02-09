from collections import defaultdict
import msgpack

class BPETokenizer:
    def __init__(self, corpus_path, sft_message_path, tokenizer_path, min_max_freq=5, max_token_len=15):
        self.max_token_len = max_token_len

        self.stoi, self.itos, self.merge_list = self._train_bpe(corpus_path, sft_message_path, min_max_freq)

        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.special_tokens = [
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.pad_token
        ]
        self._set_special_token(self.unk_token)
        self._set_special_token(self.bos_token)
        self._set_special_token(self.eos_token)
        self._set_special_token(self.pad_token)
        self.vocab_size = len(self.stoi)

        self.save(tokenizer_path)

    def _train_bpe(self, corpus_path, sft_message_path, min_max_freq):
        merge_list = []
        texts = []
        text_set = set()
        with open(corpus_path, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line_set = set(line)
                text_set.update(line_set)
                texts.append(line)
        with open(sft_message_path, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line_set = set(line)
                text_set.update(line_set)
                texts.append(line)
        text_char = sorted(list(text_set))
        stoi = {ch:i for i, ch in enumerate(text_char)}
        itos = {i:ch for ch, i in stoi.items()}
        for i in range(len(texts)):
            line = texts[i]
            text = []
            for ch in line:
                text.append(stoi[ch])
            texts[i] = text
        while True:
            pair_freq = defaultdict(int)
            for text in texts:
                for i in range(len(text)-1):
                    pair = (text[i], text[i+1])
                    pair_freq[pair] += 1
            max_freq_pair, max_freq = max(pair_freq.items(), key=lambda x: x[1])
            if max_freq < min_max_freq:
                break
            max_freq_pair_i = len(stoi)
            max_freq_pair_s = itos[max_freq_pair[0]] + itos[max_freq_pair[1]]
            stoi[max_freq_pair_s] = max_freq_pair_i
            itos[max_freq_pair_i] = max_freq_pair_s
            merge_list.append((max_freq_pair[0], max_freq_pair[1], max_freq_pair_i))
            for i in range(len(texts)):
                new_text = []
                text = texts[i]
                j = 0
                while j < len(text)-1:
                    if text[j] == max_freq_pair[0] and text[j+1] == max_freq_pair[1]:
                        new_text.append(max_freq_pair_i)
                        j += 1
                    else:
                        new_text.append(text[j])
                    j += 1
                texts[i] = new_text
        return stoi, itos, merge_list

    def _set_special_token(self, token):
        i = len(self.stoi)
        self.stoi[token] = i
        self.itos[i] = token

    def encode(self, text, add_bos=True, add_eos=True):
        ids = []
        if add_bos:
            ids.append(self.stoi[self.bos_token])
        old_ids = []
        for ch in text:
            old_ids.append(self.stoi[ch])
        new_ids = []
        for merge in self.merge_list:
            new_ids = []
            pair_a = merge[0]
            pair_b = merge[1]
            i = 0
            while i < len(old_ids):
                if i+1 < len(old_ids) and old_ids[i] == pair_a and old_ids[i+1] == pair_b:
                    new_ids.append(merge[2])
                    i += 1
                else:
                    new_ids.append(old_ids[i])
                i += 1
            old_ids = new_ids
        ids.extend(new_ids)
        if add_eos:
            ids.append(self.stoi[self.eos_token])
        return ids
    
    def decode(self, ids):
        chars = []
        for i in ids:
            char = self.itos[i]
            if char in self.special_tokens:
                continue
            chars.append(char)
        return "".join(chars)
    
    def save(self, path):
        with open(path, "wb") as f:
            msgpack.pack({
                "itos": self.itos,
                "stoi": self.stoi,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "special_tokens": self.special_tokens,
                "vocab_size": self.vocab_size,
                "max_token_len": self.max_token_len,
                "merge_list": self.merge_list
            }, f, use_bin_type=True)

    @classmethod
    def _from_data(cls, itos, stoi, bos_token, eos_token, pad_token, unk_token, special_tokens, vocab_size, max_token_len, merge_list):
        obj = cls.__new__(cls)
        obj.itos = itos
        obj.stoi = stoi
        obj.bos_token = bos_token
        obj.eos_token = eos_token
        obj.pad_token = pad_token
        obj.unk_token = unk_token
        obj.special_tokens = special_tokens
        obj.vocab_size = vocab_size
        obj.max_token_len = max_token_len
        obj.merge_list = merge_list
        return obj

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = msgpack.unpack(f, raw=False, strict_map_key=False)
        return cls._from_data(
            itos = data["itos"],
            stoi = data["stoi"],
            bos_token = data["bos_token"],
            eos_token = data["eos_token"],
            pad_token = data["pad_token"],
            unk_token = data["unk_token"],
            special_tokens = data["special_tokens"],
            vocab_size = data["vocab_size"],
            max_token_len = data["max_token_len"],
            merge_list = data["merge_list"]
        )