from typing import Dict, Optional

from ...text import TextPreprocessor
from .text.indexing import TacotronIndexer
from .text.normalization import TacotronEnglishCleaner
from .text.tokenization import TacotronTextTokenizer

__all__ = [
    "TacotronEnglishTextPreprocessor",
    "TacotronEnglishCleaner",
    "TacotronTextTokenizer",
    "TacotronIndexer",
]


class TacotronEnglishTextPreprocessor(TextPreprocessor):
    def __init__(
        self,
        abbreviations: Optional[Dict[str, str]] = None,
    ) -> None:
        normalizer = TacotronEnglishCleaner(abbreviations)
        tokenizer = TacotronTextTokenizer()
        indexer = TacotronIndexer()

        super().__init__(
            normalizer=normalizer,
            tokenizer=tokenizer,
            phonemizer=None,
            indexer=indexer,
        )
