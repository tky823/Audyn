import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional.bitnet import round_clip

__all__ = [
    "BitLinearB158",
    "RoundClip",
]


class BitLinearB158(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        weight = torch.empty(
            (out_features, in_features),
            **factory_kwargs,
        )
        weight = nn.Parameter(weight, requires_grad=True)
        self.register_parameter("weight", weight)

        self.bits = bits
        self.eps = eps

        self._reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bits = self.bits
        eps = self.eps

        # quantize input
        q = 2**bits
        abs_input = torch.abs(input)
        gamma = torch.max(abs_input)
        gamma = torch.clamp(gamma, min=eps)
        x = input * q / gamma
        x = round_clip(x, min=-q, max=q - 1)

        # quantize weight
        abs_weight = torch.abs(self.weight)
        beta = torch.mean(abs_weight)
        beta = torch.clamp(beta, min=eps)
        weight = abs_weight / beta
        quantized_weight = round_clip(weight, min=-1, max=1)

        x = F.linear(x, quantized_weight)
        output = x * (beta * gamma) / q

        return output

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class RoundClip(nn.Module):
    def __init__(self, min: float = -1, max: float = 1) -> None:
        super().__init__()

        self.min = min
        self.max = max

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = torch.round(input)
        x = torch.clamp(x, min=self.min, max=self.max)
        output = torch.detach(x - input) + input

        return output
