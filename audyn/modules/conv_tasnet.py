from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t

from .tasnet import get_layer_norm

__all__ = [
    "TimeDilatedConvNet",
    "TimeDilatedConvBlock1d",
    "ResidualBlock1d",
    "DepthwiseSeparableConv1d",
]


class TimeDilatedConvNet(nn.Module):
    """Backbone network of Conv-TasNet."""

    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 256,
        skip_channels: int = 256,
        kernel_size: _size_1_t = 3,
        num_blocks: int = 3,
        num_layers: int = 10,
        dilated: bool = True,
        separable: bool = False,
        is_causal: bool = True,
        nonlinear: Optional[str] = None,
        norm: Optional[Union[bool, str]] = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.num_blocks = num_blocks

        net = []

        for idx in range(num_blocks):
            if idx == num_blocks - 1:
                net.append(
                    TimeDilatedConvBlock1d(
                        num_features,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        num_layers=num_layers,
                        dilated=dilated,
                        separable=separable,
                        is_causal=is_causal,
                        nonlinear=nonlinear,
                        norm=norm,
                        dual_head=False,
                        eps=eps,
                    )
                )
            else:
                net.append(
                    TimeDilatedConvBlock1d(
                        num_features,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        num_layers=num_layers,
                        dilated=dilated,
                        separable=separable,
                        is_causal=is_causal,
                        nonlinear=nonlinear,
                        norm=norm,
                        dual_head=True,
                        eps=eps,
                    )
                )

        self.net = nn.Sequential(*net)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        num_blocks = self.num_blocks

        x = input
        skip_connection = 0

        for idx in range(num_blocks):
            x, skip = self.net[idx](x)
            skip_connection = skip_connection + skip

        output = skip_connection

        return output


class TimeDilatedConvBlock1d(nn.Module):
    """Time-dilated convolution block for Conv-TasNet."""

    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 256,
        skip_channels: int = 256,
        kernel_size: _size_1_t = 3,
        num_layers: int = 10,
        dilated: bool = True,
        separable: bool = False,
        is_causal: bool = True,
        nonlinear: Optional[str] = None,
        norm: Optional[Union[bool, str]] = True,
        dual_head: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers

        net = []

        for idx in range(num_layers):
            if dilated:
                dilation = 2**idx
                stride = 1
            else:
                dilation = 1
                stride = 2

            if not dual_head and idx == num_layers - 1:
                _dual_head = False
            else:
                _dual_head = True

            net.append(
                ResidualBlock1d(
                    num_features,
                    hidden_channels=hidden_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    separable=separable,
                    is_causal=is_causal,
                    nonlinear=nonlinear,
                    norm=norm,
                    dual_head=_dual_head,
                    eps=eps,
                )
            )

        self.net = nn.Sequential(*net)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_layers = self.num_layers

        x = input
        skip_connection = 0

        for idx in range(num_layers):
            x, skip = self.net[idx](x)
            skip_connection = skip_connection + skip

        return x, skip_connection


class ResidualBlock1d(nn.Module):
    """Residual block for Conv-TasNet."""

    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 256,
        skip_channels: int = 256,
        kernel_size: _size_1_t = 3,
        stride: _size_1_t = 2,
        dilation: _size_1_t = 1,
        separable: bool = False,
        is_causal: bool = True,
        nonlinear: Optional[str] = None,
        norm: Optional[Union[bool, str]] = True,
        dual_head: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.separable, self.is_causal = separable, is_causal
        self.norm = norm
        self.dual_head = dual_head

        self.bottleneck_conv1d = nn.Conv1d(num_features, hidden_channels, kernel_size=1, stride=1)

        if nonlinear is not None:
            if nonlinear == "prelu":
                self.nonlinear = nn.PReLU()
            else:
                raise ValueError("{} is not supported.".format(nonlinear))
        else:
            self.nonlinear = None

        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            norm_type = norm
            self.norm = get_layer_norm(norm_type, hidden_channels, eps=eps)
        elif isinstance(norm, bool):
            if norm:
                norm_type = "cLN" if is_causal else "gLN"
                self.norm = get_layer_norm(
                    norm_type, hidden_channels, is_causal=is_causal, eps=eps
                )
            else:
                self.norm = None
        else:
            raise ValueError(f"{type(norm)} is not supported as norm.")

        if separable:
            self.separable_conv1d = DepthwiseSeparableConv1d(
                hidden_channels,
                num_features,
                skip_channels=skip_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                nonlinear=nonlinear,
                norm=norm,
                dual_head=dual_head,
                eps=eps,
            )
        else:
            if dual_head:
                self.output_conv1d = nn.Conv1d(
                    hidden_channels, num_features, kernel_size=kernel_size, dilation=dilation
                )

            self.skip_conv1d = nn.Conv1d(
                hidden_channels, skip_channels, kernel_size=kernel_size, dilation=dilation
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        nonlinear, norm = self.nonlinear, self.norm
        separable, is_causal = self.separable, self.is_causal
        dual_head = self.dual_head

        length = input.size(-1)

        residual = input
        x = self.bottleneck_conv1d(input)

        if nonlinear is not None:
            x = nonlinear(x)

        if norm is not None:
            x = norm(x)

        padding = (length - 1) * stride - length + (kernel_size - 1) * dilation + 1

        if is_causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        x = F.pad(x, (padding_left, padding_right))

        if separable:
            output, skip = self.separable_conv1d(x)  # output may be None
        else:
            if dual_head:
                output = self.output_conv1d(x)
            else:
                output = None

            skip = self.skip_conv1d(x)

        if output is not None:
            output = output + residual

        return output, skip


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for Conv-TasNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        skip_channels: int = 256,
        kernel_size: _size_1_t = 3,
        stride: _size_1_t = 2,
        dilation: _size_1_t = 1,
        is_causal: bool = True,
        nonlinear: Optional[str] = None,
        norm: Optional[Union[bool, str]] = True,
        dual_head: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.dual_head = dual_head
        self.norm = norm
        self.eps = eps

        self.depthwise_conv1d = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
        )

        if nonlinear is not None:
            if nonlinear == "prelu":
                self.nonlinear = nn.PReLU()
            else:
                raise ValueError("{} is not supported.".format(nonlinear))
        else:
            self.nonlinear = None

        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            norm_type = norm
            self.norm = get_layer_norm(norm_type, in_channels, eps=eps)
        elif isinstance(norm, bool):
            if norm:
                norm_type = "cLN" if is_causal else "gLN"
                self.norm = get_layer_norm(norm_type, in_channels, is_causal=is_causal, eps=eps)
            else:
                self.norm = None
        else:
            raise ValueError(f"{type(norm)} is not supported as norm.")

        if dual_head:
            self.output_pointwise_conv1d = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1
            )
        else:
            self.output_pointwise_conv1d = None

        self.skip_pointwise_conv1d = nn.Conv1d(in_channels, skip_channels, kernel_size=1, stride=1)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nonlinear = self.nonlinear
        norm = self.norm
        dual_head = self.dual_head

        x = self.depthwise_conv1d(input)

        if nonlinear is not None:
            x = self.nonlinear(x)

        if norm is not None:
            x = norm(x)

        if dual_head:
            output = self.output_pointwise_conv1d(x)
        else:
            output = None

        skip = self.skip_pointwise_conv1d(x)

        return output, skip
