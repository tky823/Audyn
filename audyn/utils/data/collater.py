from abc import abstractmethod
from typing import Any, List

__all__ = [
    "BaseCollater",
]


class BaseCollater:
    """Base class of collater."""

    @abstractmethod
    def __call__(self, batch: List[Any]) -> None:
        pass
