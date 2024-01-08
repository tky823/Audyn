from collections import OrderedDict
from typing import List

from torchtext.vocab import vocab as build_vocab

from ...text.indexing import BaseTextIndexer
from . import SPECIALS
from . import full_symbols as cmudict_full_symbols

__all__ = ["CMUDictIndexer"]


class CMUDictIndexer(BaseTextIndexer):
    def __init__(self) -> None:
        super().__init__()

        table = []

        # To be compatible with order of full_symbols,
        # - set start=1
        # - set special_first=False.
        for idx, symbol in enumerate(cmudict_full_symbols, start=1):
            table.append((symbol, idx))

        self.vocab = build_vocab(OrderedDict(table), specials=SPECIALS, special_first=False)

    def index(self, phonemes: List[str]) -> List[int]:
        """Map each phoneme to corresponding index.

        Args:
            phonemes (list): Phoneme sequence.

        Returns:
            list: Indexed phonemes.

        """
        indices = self.vocab(phonemes)

        return indices
