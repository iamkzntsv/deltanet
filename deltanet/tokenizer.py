from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str):
        pass

    def decode(self, toks: str):
        pass


class CharTokenizer(Tokenizer):
    def __init__(self, corpus: str):
        """Simple character tokenizer"""
        with open(corpus, encoding="utf-8") as f:
            corpus = f.read()

        chars = sorted(set(corpus))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.vocab_size = len(self.stoi)

    def get_vocab_size(self):
        return self.vocab_size

    def encode(self, text: str):
        toks = []
        for ch in text:
            try:
                if ch in self.stoi:
                    toks.append(self.stoi.get(ch))
            except KeyError:
                raise ValueError("Non-existing token id encountered") from None

        return [self.stoi.get(ch, -1) for ch in text]

    def decode(self, toks: str):
        return [self.itos[tok] for tok in toks]
