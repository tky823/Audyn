import re
from abc import ABC, abstractmethod
from typing import List

_puctuation_split_re = re.compile(r"\b(?:\w+(?:'\w+)?)\b|[,.!?:;\(\)]\s*")
_whitespace_split_re = re.compile(r"\s+")


class BaseTextTokenizer(ABC):
    """Base class of text tokenizer."""

    def __call__(self, *args, **kwargs) -> List[str]:
        """Tokenize (normalized) text by calling ``self.tokenize`` method.

        Args:
            text (str): Text to be tokenized.

        Returns:
            list: Tokenized text.

        """
        return self.tokenize(*args, **kwargs)

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize (normalized) text.

        Args:
            text (str): Text to be tokenized.

        Returns:
            list: Tokenized text.

        """
        pass


class EnglishWordTokenizer(BaseTextTokenizer):
    """Tokenizer that split English sentence into words by white space."""

    def tokenize(self, text: str) -> List[str]:
        """Tokenize english sentence into words.

        .. example::

            >>> tokenizer = EnglishWordTokenizer()
            >>> text = "It's the ghost!"
            >>> tokenizer(text)
            ["It's", 'the', 'ghost', '!']

        """
        tokens = []

        for token in _whitespace_split_re.split(text):
            tokens = tokens + _puctuation_split_re.findall(token)

        return tokens
