from typing import Optional, Union

import torch
import torch.distributed as dist

from .base import BaseMetricWrapper, MultiMetrics, StatefulMetric

__all__ = ["StatefulMetric", "BaseMetricWrapper", "MultiMetrics", "MeanMetric"]


class MeanMetric(StatefulMetric):
    """Compute mean.

    .. note::

        This class supports distributed data parallel.

    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__(device=device)

        self.reset()

    def reset(self) -> None:
        self.num_samples = 0
        self.sum_value = 0

    @torch.no_grad()
    def update(self, value: Union[int, float, torch.Tensor]) -> None:
        is_distributed = dist.is_available() and dist.is_initialized()

        if is_distributed:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        if isinstance(value, torch.Tensor):
            assert value.numel() == 1, "Only scaler is supported."

            value = value.detach().item()
        elif isinstance(value, (int, float)):
            pass
        else:
            raise ValueError(f"{type(value)} is not supported.")

        if is_distributed:
            tensor = torch.tensor(value, device=self.device)
            gathered_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered_tensor, tensor)
            gathered_tensor = torch.stack(gathered_tensor, dim=0)
            gathered_tensor = gathered_tensor.sum(dim=0)
            value = gathered_tensor.item()

        self.num_samples = self.num_samples + world_size
        self.sum_value = self.sum_value + value

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        mean = torch.tensor(self.sum_value / self.num_samples, device=self.device)

        return mean
