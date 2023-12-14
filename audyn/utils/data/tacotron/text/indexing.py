from collections import OrderedDict
from typing import List

from torchtext.vocab import vocab as build_vocab

from ....text.indexing import BaseTextIndexer
from .symbols import PAD_SYMBOL, SPECIAL_SYMBOL, full_symbols


class TacotronIndexer(BaseTextIndexer):
    def __init__(self) -> None:
        super().__init__()

        table = []

        for idx, symbol in enumerate(full_symbols):
            table.append((symbol, idx))

        self.vocab = build_vocab(OrderedDict(table), specials=[PAD_SYMBOL, SPECIAL_SYMBOL])

    def index(self, phonemes: List[str]) -> List[int]:
        """Map each phoneme to corresponding index.

        Args:
            phonemes (list): Phoneme sequence.

        Returns:
            list: Indexed phonemes.

        """
        indices = self.vocab(phonemes)

        return indices
