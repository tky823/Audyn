from abc import ABC, abstractmethod


class BaseTextNormalizer(ABC):
    """Base class of text normalizer."""

    @abstractmethod
    def __call__(self, text: str) -> str:
        """Normalize text.

        Args:
            text (str): Text to be normalized.

        Returns:
            str: Normalized text.

        """
        pass
