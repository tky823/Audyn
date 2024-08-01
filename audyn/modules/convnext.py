"""ConvNext-related modules.

The main purpose of the implementation is Vocos and WaveNext.
"""

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single

from ..models.roformer import _get_activation

__all__ = [
    "ConvNeXtBlock",
]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: _size_1_t,
        norm: nn.Module,
        activation: Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]] = "gelu",
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()

        (kernel_size,) = _single(kernel_size)

        assert kernel_size % 2 == 1, "Only odd number is supported as kernel_size."

        self.kernel_size = _single(kernel_size)

        self.depthwise_conv1d = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
        )
        self.norm = norm
        self.pointwise_conv1d_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        if isinstance(activation, str):
            activation = _get_activation(activation)

        self.activation = activation
        self.pointwise_conv1d_out = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

        if scale is None:
            self.register_parameter("scale", None)
        else:
            scale = torch.full((), fill_value=scale)
            self.scale = nn.Parameter(scale, requires_grad=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        (kernel_size,) = self.kernel_size
        padding = kernel_size // 2

        residual = input
        x = F.pad(input, (padding, padding))
        x = self.depthwise_conv1d(x)
        x = x.transpose(-2, -1)
        x = self.norm(x)
        x = x.transpose(-2, -1)
        x = self.pointwise_conv1d_in(x)
        x = self.activation(x)
        x = self.pointwise_conv1d_out(x)

        if self.scale is not None:
            x = self.scale * x

        output = x + residual

        return output
