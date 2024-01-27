import os
from typing import List, Optional

from torchtext.vocab import build_vocab_from_iterator

from .....utils import audyn_cache_dir
from ....text.indexing import BaseTextIndexer

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

    def index(self, text: List[str]) -> List[int]:
        tokens = self.vocab(text)

        return tokens

    @staticmethod
    def build_vocab(path: str) -> List[str]:
        with open(path) as f:
            for line in f:
                yield [line.strip()]
