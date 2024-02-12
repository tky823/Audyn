from typing import List

from transformers import BertTokenizer as _BERTTokenizer

from audyn.utils.text import TextPreprocessor
from audyn.utils.text.indexing import BaseTextIndexer
from audyn.utils.text.tokenization import BaseTextTokenizer


class BERTTextPreprocessor(TextPreprocessor):
    def __init__(self, pretrained_name_or_path: str) -> None:
        normalizer = None
        tokenizer = BERTTokenizer(pretrained_name_or_path)
        phonemizer = None
        indexer = BERTIndexer(pretrained_name_or_path)

        super().__init__(
            normalizer=normalizer,
            tokenizer=tokenizer,
            phonemizer=phonemizer,
            indexer=indexer,
        )


class BERTTokenizer(BaseTextTokenizer):
    def __init__(self, pretrained_name_or_path: str) -> None:
        super().__init__()

        self.tokenizer = _BERTTokenizer.from_pretrained(pretrained_name_or_path)

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)


class BERTIndexer(BaseTextIndexer):
    """Indexer for BERT.

    .. note::

        This indexer prepends [CLS] token by default.

    """

    def __init__(self, pretrained_name_or_path: str) -> None:
        super().__init__()

        self.tokenizer = _BERTTokenizer.from_pretrained(pretrained_name_or_path)

    def index(self, phonemes: List[str]) -> List[int]:
        indices = self.tokenizer.convert_tokens_to_ids(phonemes)
        indices = self.tokenizer.build_inputs_with_special_tokens(indices)

        return indices
