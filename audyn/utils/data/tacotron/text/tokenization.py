from typing import List

from ....text.tokenization import BaseTextTokenizer
from ....text.vocab import Vocab
from .symbols import PAD_SYMBOL, SPECIAL_SYMBOL, full_symbols


class TacotronTextTokenizer(BaseTextTokenizer):
    def __init__(self) -> None:
        self.vocab = Vocab()

        specials = [PAD_SYMBOL, SPECIAL_SYMBOL]
        vocab_idx = 0

        for symbol in specials:
            self.vocab[symbol] = vocab_idx
            vocab_idx += 1

        for symbol in full_symbols:
            if symbol in specials:
                continue

            self.vocab[symbol] = vocab_idx
            vocab_idx += 1

    def tokenize(self, text: str) -> List[str]:
        tokens = []

        for s in text:
            if s in self.vocab and s != PAD_SYMBOL and s != "~":
                tokens.append(s)

        return tokens
