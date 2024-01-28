from typing import Optional

from ...text import TextPreprocessor
from .text.indexing import ClothoTextIndexer
from .text.normalization import ClothoTextNormalizer
from .text.symbols import vocab_size
from .text.tokenization import ClothoTextTokenizer

__all__ = ["vocab_size", "ClothoTextNormalizer", "ClothoTextTokenizer", "ClothoTextIndexer"]


class ClothoTextPreprocessor(TextPreprocessor):
    def __init__(self, root: Optional[str] = None) -> None:
        normalizer = ClothoTextNormalizer()
        tokenizer = ClothoTextTokenizer()
        indexer = ClothoTextIndexer(root)

        super().__init__(
            normalizer=normalizer,
            tokenizer=tokenizer,
            phonemizer=None,
            indexer=indexer,
        )
