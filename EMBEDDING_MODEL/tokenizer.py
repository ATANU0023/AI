import torch

class SimpleTokenizer:
    def __init__(self, sentences=None):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab_size = 4
        
        if sentences:
            self.build_vocab(sentences)

    def build_vocab(self, sentences):
        for sentence in sentences:
            for word in sentence.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text, max_length=10):
        tokens = text.lower().split()
        indices = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in tokens]
        indices = [self.word2idx["<SOS>"]] + indices + [self.word2idx["<EOS>"]]
        
        # Padding
        if len(indices) < max_length:
            indices += [self.word2idx["<PAD>"]] * (max_length - len(indices))
        else:
            indices = indices[:max_length-1] + [self.word2idx["<EOS>"]]
            
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        words = []
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
            
        for idx in indices:
            word = self.idx2word.get(idx, "<UNK>")
            if word == "<EOS>":
                break
            if word not in ["<SOS>", "<PAD>"]:
                words.append(word)
        return " ".join(words)
