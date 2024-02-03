from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

__all__ = ["StatefulMetric", "BaseMetricWrapper"]


class StatefulMetric(ABC):
    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__()

        self.device = device

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Update state of metric."""

    @abstractmethod
    def reset(self) -> None:
        """Reset states."""

    @abstractmethod
    def compute(self) -> Any:
        """Compute metric by current states."""

    def to(self, device: torch.device) -> "StatefulMetric":
        """Change device parameter."""
        self.device = device


class BaseMetricWrapper(StatefulMetric):
    """Wrapper class to handle multiple metrics by MultiMetrics."""

    def __init__(
        self,
        metric: StatefulMetric,
        key_mapping: Dict[str, Any],
        weight: float = 1,
    ) -> None:
        super().__init__()

        self.metric = metric
        self.key_mapping = key_mapping
        self.weight = weight

    def reset(self) -> None:
        """Reset states."""

    def update(self, *args, **kwargs) -> Any:
        """Update state of metric."""
        self.metric.update(*args, **kwargs)

    def compute(self, *args, **kwargs) -> Any:
        """Compute metric by current states."""
        self.metric.compute(*args, **kwargs)

    def to(self, device: torch.device) -> StatefulMetric:
        """Change device parameter."""
        self.metric.to(device=device)
