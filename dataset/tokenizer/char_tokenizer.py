class CharTokenizer:
    def __init__(self, corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(list(set(text)))

        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        self.special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token
        ]

        self.vocab = self.special_tokens + chars
        self.vocab_size = len(self.vocab)

        self.stoi = {ch:i for i, ch in enumerate(self.vocab)}
        self.itos = {i:ch for ch, i in self.stoi.items()}

    def encode(self, text, add_bos=True, add_eos=True):
        ids = []
        if add_bos:
            ids.append(self.stoi[self.bos_token])
        for char in text:
            if char in self.stoi:
                ids.append(self.stoi[char])
            else:
                ids.append(self.stoi[self.unk_token])
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
