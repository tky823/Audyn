"""ConvNext-related modules.

The main purpose of the implementation is Vocos and WaveNext.
"""

import copy
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single

from ..models.roformer import _get_activation

__all__ = [
    "StackedConvNeXtBlock1d",
    "ConvNeXtBlock1d",
]


class StackedConvNeXtBlock1d(nn.Module):
    """Stacked ConvNeXt blocks.

    Args:
        num_features (int): Number of features.
        hidden_channels (int): Number of hidden channels.
        kernel_size (_size_1_t): Kernel size in depthwise convolutions.
        norm (nn.Module): Layer normalization module, which takes
            (batch_size, length, num_features) such as ``nn.LayerNorm``.
        activation (str, nn.Module or callable): Activation module.
        num_blocks (int): Number of blocks. This parameter also defines the output scale
            of each block. Default: ``12``.

    """

    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        kernel_size: Union[List[_size_1_t], _size_1_t],
        norm: nn.Module,
        activation: Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]] = "gelu",
        num_blocks: int = 12,
    ) -> None:
        super().__init__()

        self.num_blocks = num_blocks

        if type(kernel_size) in [int, tuple]:
            kernel_size = [kernel_size] * num_blocks
        else:
            assert len(kernel_size) == num_blocks

        backbone = []
        _scale = 1 / num_blocks

        for block_idx in range(num_blocks):
            _kernel_size = kernel_size[block_idx]
            _norm = copy.deepcopy(norm)
            _activation = copy.deepcopy(activation)
            block = ConvNeXtBlock1d(
                num_features,
                hidden_channels,
                kernel_size=_kernel_size,
                norm=_norm,
                activation=_activation,
                scale=_scale,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of StackedConvNeXtBlock1d.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, num_features, length).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, num_features, length).

        """
        num_blocks = self.num_blocks

        x = input

        for stack_idx in range(num_blocks):
            x = self.backbone[stack_idx](x)

        output = x

        return output


class ConvNeXtBlock1d(nn.Module):
    """ConvNeXt block.

    Args:
        num_features (int): Number of features.
        hidden_channels (int): Number of hidden channels.
        kernel_size (_size_1_t): Kernel size in depthwise convolutions.
        norm (nn.Module): Layer normalization module, which takes
            (batch_size, length, num_features) such as ``nn.LayerNorm``.
        activation (str, nn.Module or callable): Activation module.
        scale (float, optional): Initial scale of output.

    """

    def __init__(
        self,
        num_features: int,
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
            num_features,
            num_features,
            kernel_size=kernel_size,
            groups=num_features,
        )
        self.norm = norm
        self.pointwise_conv1d_in = nn.Conv1d(num_features, hidden_channels, kernel_size=1)

        if isinstance(activation, str):
            activation = _get_activation(activation)

        self.activation = activation
        self.pointwise_conv1d_out = nn.Conv1d(hidden_channels, num_features, kernel_size=1)

        if scale is None:
            self.register_parameter("scale", None)
        else:
            scale = torch.full((), fill_value=scale)
            self.scale = nn.Parameter(scale, requires_grad=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of ConvNeXtBlock1d.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, num_features, length).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, num_features, length).

        """
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
