import math
from typing import Optional

import torch
import torch.nn as nn

from ..functional.bitnet import bit_linear_b158

__all__ = [
    "BitLinearB158",
    "RoundClip",
]


class BitLinearB158(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        bias: bool = True,
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

        if bias:
            bias = torch.empty(
                (out_features,),
                **factory_kwargs,
            )
            bias = nn.Parameter(bias, requires_grad=True)
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.eps = eps

        self._reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = bit_linear_b158(
            input,
            self.weight,
            bias=self.bias,
            bits=self.bits,
            eps=self.eps,
        )

        return output

    def _reset_parameters(self) -> None:
        # ported from https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L105-L113  # noqa: E501
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        in_features = self.in_features
        out_features = self.out_features
        bits = self.bits
        bias = self.bias is not None

        return f"in_features={in_features}, out_features={out_features}, bits={bits}, bias={bias}"


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
