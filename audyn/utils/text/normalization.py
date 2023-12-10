from abc import ABC, abstractmethod


class BaseTextNormalizer(ABC):
    """Base class of text normalizer."""

    def __call__(self, *args, **kwargs) -> str:
        """Normalize text."""
        return self.normalize(*args, **kwargs)

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize text.

        Args:
            text (str): Text to be normalized.

        Returns:
            str: Normalized text.

        """
        pass
