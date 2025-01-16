from typing import List

from ....text.indexing import BaseTextIndexer
from ....text.vocab import Vocab
from .symbols import PAD_SYMBOL, SPECIAL_SYMBOL, full_symbols


class TacotronIndexer(BaseTextIndexer):
    def __init__(self) -> None:
        super().__init__()

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

    def index(self, phonemes: List[str]) -> List[int]:
        """Map each phoneme to corresponding index.

        Args:
            phonemes (list): Phoneme sequence.

        Returns:
            list: Indexed phonemes.

        """
        indices = self.vocab(phonemes)

        return indices
