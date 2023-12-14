from abc import ABC, abstractmethod
from typing import List


class BaseTextIndexer(ABC):
    """Base class to index phoneme."""

    def __call__(self, *args, **kwargs) -> str:
        return self.index(*args, **kwargs)

    @abstractmethod
    def index(self, phonemes: List[str]) -> List[int]:
        """Map each phoneme to corresponding index.

        Args:
            phonemes (list): Phoneme sequence.

        Returns:
            list: Indexed phonemes.

        """
        pass
