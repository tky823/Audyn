from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

__all__ = ["StatefulMetric"]


class StatefulMetric(ABC):
    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__()

        self.device = device

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Update state of metric."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset states."""

    def to(self, device: torch.device) -> "StatefulMetric":
        """Change device parameter."""
        self.device = device
