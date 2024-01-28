import os
from typing import List, Optional

from torchtext.vocab import build_vocab_from_iterator

from .....utils import audyn_cache_dir
from ....text.indexing import BaseTextIndexer
from .symbols import BOS_SYMBOL, EOS_SYMBOL, vocab_size

__all__ = ["ClothoTextIndexer"]


class ClothoTextIndexer(BaseTextIndexer):
    """Text indexer for Clotho dataset."""

    filename = "vocab.txt"

    def __init__(self, root: Optional[str] = None) -> None:
        super().__init__()

        if root is None:
            root = os.path.join(audyn_cache_dir, "data", "clotho")

        path = os.path.join(root, self.filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")

        self.vocab = build_vocab_from_iterator(self.build_vocab(path))

        assert (
            len(self.vocab) == vocab_size
        ), f"Vocab size is expected {vocab_size}, but {len(self.vocab)} is given."

    def index(
        self,
        phonemes: List[str],
        insert_bos_token: bool = True,
        insert_eos_token: bool = True,
    ) -> List[int]:
        """Convert text tokens into sequence of indices.

        Args:
            phonemes (list): Text tokens. Each item is ``str``.
            insert_bos_token (bool): If ``True``, ``BOS_SYMBOL`` is prepended to sequence.
            insert_eos_token (bool): If ``True``, ``EOS_SYMBOL`` is appended to sequence.

        Returns:
            list: List of indices.

        .. note::

            In terms of compatibility of ``BaseTextIndexer`` takes text tokens as ``phonemes``.

        """
        if insert_bos_token and phonemes[0] != BOS_SYMBOL:
            phonemes = [BOS_SYMBOL] + phonemes

        if insert_eos_token and phonemes[-1] != EOS_SYMBOL:
            phonemes = phonemes + [BOS_SYMBOL]

        phonemes = self.vocab(phonemes)

        return phonemes

    @staticmethod
    def build_vocab(path: str) -> List[str]:
        with open(path) as f:
            for line in f:
                yield [line.strip()]
