from typing import Optional, Union

import torch

from .base import StatefulMetric

__all__ = ["MeanMetric"]


class MeanMetric(StatefulMetric):
    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__(device=device)

        self.reset()

    def reset(self) -> None:
        self.num_samples = 0
        self.sum_value = 0

    def update(self, value: Union[int, float, torch.Tensor]) -> None:
        if isinstance(value, torch.Tensor):
            assert value.numel() == 1, "Only scaler is supported."

            value = value.detach().item()
        elif isinstance(value, (int, float)):
            pass
        else:
            raise ValueError(f"{type(value)} is not supported.")

        self.num_samples = self.num_samples + 1
        self.sum_value = self.sum_value + value

    def compute(self) -> torch.Tensor:
        mean = torch.tensor(self.sum_value / self.num_samples, device=self.device)

        return mean
