from abc import ABC, abstractmethod
from typing import Any

__all__ = ["StatefulMetric"]


class StatefulMetric(ABC):
    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Update state of metric."""
        pass
