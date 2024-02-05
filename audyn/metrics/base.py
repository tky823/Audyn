from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

__all__ = ["StatefulMetric", "BaseMetricWrapper", "MultiMetrics"]


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
        return self.metric.compute(*args, **kwargs)

    def to(self, device: torch.device) -> StatefulMetric:
        """Change device parameter."""
        self.metric.to(device=device)


class MultiMetrics(StatefulMetric):
    """Base class of dict-type multiple metrics."""

    def __init__(self, device: Optional[torch.device] = None, **kwargs) -> None:
        super().__init__(device=device)

        self.metrics: Dict[str, StatefulMetric] = {}

        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            assert isinstance(k, str), f"Invalid key {k} is found."
            assert isinstance(v, StatefulMetric)

            if v.device is not None:
                v = v.to(device)

            self.metrics[k] = v

    def __getitem__(self, __key: str) -> StatefulMetric:
        return self.metrics[__key]

    def reset(self, *args, **kwargs) -> None:
        for k in sorted(self.metrics.keys()):
            self.metrics[k].reset(*args, **kwargs)

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError("update is not supported.")

    def compute(self, *args, **kwargs) -> None:
        raise NotImplementedError("compute is not supported.")

    def to(self, device: Optional[torch.device] = None) -> "MultiMetrics":
        for k in sorted(self.metrics.keys()):
            self.metrics[k].to(device)

        return self
