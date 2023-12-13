from collections import OrderedDict
from typing import List

from torchtext.vocab import vocab as build_vocab

from ....text.tokenization import BaseTextTokenizer
from .symbols import PAD_SYMBOL, SPECIAL_SYMBOL, full_symbols


class TacotronTextTokenizer(BaseTextTokenizer):
    def __init__(self) -> None:
        table = []

        for idx, symbol in enumerate(full_symbols):
            table.append((symbol, idx))

        self.vocab = build_vocab(OrderedDict(table), specials=[PAD_SYMBOL, SPECIAL_SYMBOL])

    def tokenize(self, text: str) -> List[str]:
        tokens = []

        for s in text:
            if s in self.vocab and s != PAD_SYMBOL and s != "~":
                tokens.append(s)

        return tokens
