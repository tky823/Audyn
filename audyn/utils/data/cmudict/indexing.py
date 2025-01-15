from typing import List

from ...text.indexing import BaseTextIndexer
from ...text.vocab import Vocab
from . import full_symbols as cmudict_full_symbols

__all__ = ["CMUDictIndexer"]


class CMUDictIndexer(BaseTextIndexer):
    def __init__(self) -> None:
        super().__init__()

        self.vocab = Vocab()

        vocab_idx = 0

        for symbol in cmudict_full_symbols:
            self.vocab[symbol] = vocab_idx
            vocab_idx += 1

    def index(self, phonemes: List[str]) -> List[int]:
        """Map each phoneme to corresponding index.

        Args:
            phonemes (list): Phoneme sequence.

        Returns:
            list: Indexed phonemes.

        """
        indices = self.vocab(phonemes)

        return indices
