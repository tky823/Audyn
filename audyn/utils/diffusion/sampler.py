import math
import warnings
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "ReverseSampler",
    "DDPMSampler",
]


class ReverseSampler(ABC):
    def __init__(self, denoiser: nn.Module) -> None:
        warnings.warn("ReverseSampler is experimental.", UserWarning, stacklevel=2)

        super().__init__()

        self.denoiser = denoiser

    @abstractmethod
    def sample(self, *args, **kwargs) -> torch.Tensor:
        pass


class DDPMSampler(ReverseSampler):
    def __init__(
        self, denoiser: nn.Module, max_step: int = 1000, std: float = 1, seed: int = 0
    ) -> None:
        super().__init__(denoiser)

        self.max_step = max_step
        self.std = std
        self.seed = seed

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def sample(self, input: torch.Tensor, *args, step: int = None, **kwargs) -> torch.Tensor:
        if step is None:
            raise ValueError("step is required.")

        factory_kwargs = {
            "dtype": input.dtype,
            "device": input.device,
        }
        noise = torch.randn(input.size(), generator=self.generator, **factory_kwargs)
        noise = self.std * noise / math.sqrt(self.max_step)

        output = self.denoiser(input, *args, step=step, **kwargs) + noise

        return output

    def initial_noise(
        self,
        size: torch.Size,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        noise = torch.randn(size, **factory_kwargs)

        return noise
