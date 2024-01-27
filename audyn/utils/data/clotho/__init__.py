from typing import Union

import torchtext

from ...text import TextPreprocessor
from .text.indexing import ClothoTextIndexer
from .text.normalization import ClothoTextNormalizer
from .text.tokenization import ClothoTextTokenizer

__all__ = ["ClothoTextNormalizer", "ClothoTextTokenizer", "ClothoTextIndexer"]


class ClothoTextPreprocessor(TextPreprocessor):
    def __init__(self, vocab: Union[str, torchtext.vocab.Vocab]) -> None:
        normalizer = ClothoTextNormalizer()
        tokenizer = ClothoTextTokenizer()
        indexer = ClothoTextIndexer(vocab)

        super().__init__(
            normalizer=normalizer,
            tokenizer=tokenizer,
            phonemizer=None,
            indexer=indexer,
        )
