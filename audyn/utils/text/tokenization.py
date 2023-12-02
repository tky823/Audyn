from abc import ABC, abstractmethod
from typing import List


class BaseTextTokenizer(ABC):
    """Base class of text tokenizer."""

    @abstractmethod
    def __call__(self, text: str) -> List[str]:
        """Tokenize (normalized) text.

        Args:
            text (str): Text to be tokenized.

        Returns:
            list: Tokenized text.

        """
        pass
