"""Modules of PixelSNAIL.

See https://arxiv.org/abs/1712.09763 for the details.
"""
import copy
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.common_types import _size_2_t

__all__ = ["ResidualBlock2d", "Conv2d", "CausalConv2d"]

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")
available_weight_regularizations = {"weight_norm", "spectral_norm"}


class ResidualBlock2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        kernel_size: _size_2_t,
        groups: int = 1,
        bias: bool = True,
        weight_regularization: Optional[str] = "weight_norm",
        activation: Optional[
            Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = "elu",
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        if isinstance(activation, str):
            activation_1 = _get_activation(activation)
            activation_2 = _get_activation(activation)
        else:
            # NOTE: Activations are not shared with each other.
            activation_1 = copy.deepcopy(activation)
            activation_2 = copy.deepcopy(activation)

        self.activation_1 = activation_1
        self.conv2d_in = CausalConv2d(
            num_features,
            num_features,
            kernel_size=kernel_size,
            groups=groups,
            bias=bias,
            **factory_kwargs,
        )
        self.activation_2 = activation_2
        self.conv2d_out = CausalConv2d(
            num_features,
            2 * num_features,
            kernel_size=kernel_size,
            groups=groups,
            bias=bias,
            **factory_kwargs,
        )
        self.glu = nn.GLU(dim=1)

        if weight_regularization is not None:
            if weight_regularization == "weight_norm":
                self.weight_norm_()
            elif weight_regularization == "spectral_norm":
                self.spectral_norm_()
            else:
                raise ValueError(
                    "{}-based weight regularization is not supported.".format(
                        weight_regularization
                    )
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of ResidualBlock2d.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, num_features, height, width).

        """
        x = self.activation_1(input)
        x = self.conv2d_in(x)
        x = self.activation_2(x)
        x = self.conv2d_out(x)
        output = input + self.glu(x)

        return output

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.conv2d_in = weight_norm_fn(self.conv2d_in)
        self.conv2d_out = weight_norm_fn(self.conv2d_out)

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.conv2d_in = remove_weight_norm_fn(self.conv2d_in, *remove_weight_norm_args)
        self.conv2d_out = remove_weight_norm_fn(self.conv2d_out, *remove_weight_norm_args)

    def spectral_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            spectral_norm_fn = nn.utils.spectral_norm
        else:
            spectral_norm_fn = nn.utils.parametrizations.spectral_norm

        self.conv2d_in = spectral_norm_fn(self.conv2d_in)
        self.conv2d_out = spectral_norm_fn(self.conv2d_out)

    def remove_spectral_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_spectral_norm_fn = nn.utils.remove_spectral_norm
            remove_spectral_norm_args = ()
        else:
            remove_spectral_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_spectral_norm_args = ("weight",)

        self.conv2d_in = remove_spectral_norm_fn(self.conv2d_in, *remove_spectral_norm_args)
        self.conv2d_out = remove_spectral_norm_fn(self.conv2d_out, *remove_spectral_norm_args)


class Conv2d(nn.Conv2d):
    """Causal convolution for PixelSNAIL."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        groups: int = 1,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of Conv2d in PixelSNAIL.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, out_channels, height, width).

        .. note::

            When ``kernel_size`` is ``(2, 3)`` and ``input.size()`` is ``(*, *, 5, 6)``,
            the point represented as 'x' in the top figure depends on the receptive field
            represented as '*' in the bottom figure.

            output:
                |-|-|-|-|-|-|
                |-|-|-|-|-|-|
                |-|-|-|-|x|-|
                |-|-|-|-|-|-|
                |-|-|-|-|-|-|

            input:
                |-|-|-|-|-|-|
                |-|-|*|*|*|-|
                |-|-|*|*|*|-|
                |-|-|-|-|-|-|
                |-|-|-|-|-|-|

        """
        kernel_height, kernel_width = self.kernel_size

        x = F.pad(input, (kernel_width - 1, 0, kernel_height - 1, 0))

        return super().forward(x)


class CausalConv2d(Conv2d):
    """Alias of Conv2d."""

    pass


def _get_activation(activation: str) -> nn.Module:
    """Get activation module by str.

    Args:
        activation (str): Name of activation module.

    Returns:
        nn.Module: Activation module.

    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "elu":
        return nn.ELU()

    raise RuntimeError(f"activation should be relu/gelu/elu, not {activation}")
