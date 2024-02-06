import os
from typing import Iterable, List, Optional

from torchtext.vocab import build_vocab_from_iterator

from .....utils import audyn_cache_dir
from .....utils.github import download_file_from_github_release
from ....text.indexing import BaseTextIndexer
from .symbols import BOS_SYMBOL, EOS_SYMBOL, MASK_SYMBOL, vocab_size

__all__ = ["ClothoTextIndexer"]


class ClothoTextIndexer(BaseTextIndexer):
    """Text indexer for Clotho dataset."""

    filename = "vocab.txt"
    chotho_vocab_urls = [
        "https://github.com/tky823/Audyn/releases/download/v0.0.1.dev1/clotho-v2-vocab.txt"
    ]

    def __init__(
        self,
        root: Optional[str] = None,
        include_mask_token: bool = False,
    ) -> None:
        super().__init__()

        if root is None:
            root = os.path.join(audyn_cache_dir, "data", "clotho")

        path = os.path.join(root, self.filename)

        if not os.path.exists(path):
            # TODO: additional sources
            assert len(self.chotho_vocab_urls) == 1

            os.makedirs(root, exist_ok=True)

            for url in self.chotho_vocab_urls:
                download_file_from_github_release(url, path)

                break

        specials = [BOS_SYMBOL, EOS_SYMBOL]

        if include_mask_token:
            specials.insert(0, MASK_SYMBOL)

        self.vocab = build_vocab_from_iterator(
            self.build_vocab(path, include_mask_token=include_mask_token),
            specials=specials,
        )

        actual_vocab_size = vocab_size + 1 if include_mask_token else vocab_size

        assert (
            len(self.vocab) == actual_vocab_size
        ), f"Vocab size is expected {actual_vocab_size}, but {len(self.vocab)} is given."

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
            phonemes = phonemes + [EOS_SYMBOL]

        phonemes = self.vocab(phonemes)

        return phonemes

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def mask_index(self) -> int:
        if MASK_SYMBOL not in self.vocab:
            raise ValueError("mask_index is not included in vocab.")

        return self.vocab[MASK_SYMBOL]

    @staticmethod
    def build_vocab(
        path: str,
        include_mask_token: bool = False,
    ) -> Iterable[List[str]]:
        with open(path) as f:
            for line in f:
                yield [line.strip()]

        if include_mask_token:
            yield [MASK_SYMBOL]
