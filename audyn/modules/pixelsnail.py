"""Modules of PixelSNAIL.

See https://arxiv.org/abs/1712.09763 for the details.
"""

import copy
import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.common_types import _size_2_t

__all__ = [
    "ResidualBlock2d",
    "Conv2d",
    "CausalConv2d",
    "PointwiseConvBlock2d",
    "CausalSelfAttention2d",
]

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")
available_weight_regularizations = {"weight_norm", "spectral_norm"}


class PixelBlock(nn.Module):
    """Block of PixelSNAIL.

    Args:
        in_channels (int): Number of input channels.
        skip_channels (int): Number of skip channels.
        kernel_size (_size_2_t): Kernel size in convolutions.
        num_heads (int): Number of heads in attention.
        num_repeats (int): Number of repeats of ``ResidualBlock2d``.
        dropout (float): Dropout rate in attention. Default: ``0.0``.
        kdim (int): Embedding dimension of keys. ``kdim`` should be divisible by ``num_heads``.
        vdim (int): Embedding dimension of values. ``vdim`` should be divisible by ``num_heads``.
        weight_regularization (str, optional): Weight regularization.
        activation (str, nn.Module, or callable): Activation function. Default: ``elu``.

    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        kernel_size: _size_2_t,
        num_heads: int,
        num_repeats: int,
        dropout: float = 0.0,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        weight_regularization: Optional[str] = "weight_norm",
        activation: Optional[
            Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = "elu",
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        if kdim is None:
            kdim = in_channels

        if vdim is None:
            vdim = in_channels

        backbone = []

        for _ in range(num_repeats):
            block = ResidualBlock2d(
                in_channels,
                kernel_size=kernel_size,
                weight_regularization=weight_regularization,
                activation=activation,
                **factory_kwargs,
            )
            backbone.append(block)

        self.backbone = nn.Sequential(*backbone)
        self.pointwise_conv2d_in = PointwiseConvBlock2d(
            in_channels,
            in_channels,
            weight_regularization=weight_regularization,
            activation=activation,
            **factory_kwargs,
        )
        self.mha2d = CausalSelfAttention2d(
            in_channels + skip_channels,
            vdim,
            kdim=kdim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.pointwise_conv2d_attn = PointwiseConvBlock2d(
            vdim,
            in_channels,
            weight_regularization=weight_regularization,
            activation=activation,
            **factory_kwargs,
        )
        self.pointwise_conv2d_out = PointwiseConvBlock2d(
            in_channels,
            in_channels,
            weight_regularization=weight_regularization,
            activation=activation,
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass of ConvBlock2d.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, height, width).
            skip (torch.Tensor): Skip feature of shape (batch_size, skip_channels, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, in_channels, height, width).

        """
        x = self.backbone(input)
        x_in = self.pointwise_conv2d_in(x)
        x_attn = torch.cat([x, skip], dim=1)
        x_attn = self.mha2d(x_attn)
        x_attn = self.pointwise_conv2d_attn(x_attn)
        x = x_in + x_attn
        output = self.pointwise_conv2d_out(x)

        return output


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


class PointwiseConvBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=groups,
            bias=bias,
            **factory_kwargs,
        )
        self.activation_2 = activation_2

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
        x = self.activation_1(input)
        x = self.conv2d(x)
        output = self.activation_2(x)

        return output

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.conv2d = weight_norm_fn(self.conv2d)

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.conv2d = remove_weight_norm_fn(self.conv2d, *remove_weight_norm_args)

    def spectral_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            spectral_norm_fn = nn.utils.spectral_norm
        else:
            spectral_norm_fn = nn.utils.parametrizations.spectral_norm

        self.conv2d = spectral_norm_fn(self.conv2d)

    def remove_spectral_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_spectral_norm_fn = nn.utils.remove_spectral_norm
            remove_spectral_norm_args = ()
        else:
            remove_spectral_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_spectral_norm_args = ("weight",)

        self.conv2d = remove_spectral_norm_fn(self.conv2d, *remove_spectral_norm_args)


class CausalSelfAttention2d(nn.Module):
    """Self-attention with causality for 2D input.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Embedding dimension of values, which is equal to number of
            output channels. ``out_channels`` should be divisible by ``num_heads``.
        kdim (int): Embedding dimension of keys. ``kdim`` should be divisible by ``num_heads``.
        num_heads (int): Number of heads in attention.
        dropout (float): Dropout rate in attention. Default: ``0.0``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kdim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.q_proj = nn.Linear(in_channels, kdim)
        self.k_proj = nn.Linear(in_channels, kdim)
        self.v_proj = nn.Linear(in_channels, out_channels)

        assert (
            out_channels % num_heads == 0
        ), f"out_channels ({out_channels}) should be divisible by num_heads ({num_heads})"
        assert (
            kdim % num_heads == 0
        ), f"kdim ({kdim}) should be divisible by num_heads ({num_heads})"

        self.out_channels = out_channels
        self.kdim = kdim
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of CausalSelfAttention2d.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, out_channels, height, width).

        """
        out_channels = self.out_channels
        kdim = self.kdim
        num_heads = self.num_heads
        dropout = self.dropout
        batch_size, in_channels, height, width = input.size()

        x = input.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, height * width, in_channels)
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        query = query.view(batch_size, height * width, num_heads, kdim // num_heads)
        key = key.view(batch_size, height * width, num_heads, kdim // num_heads)
        value = value.view(batch_size, height * width, num_heads, out_channels // num_heads)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attn_score = torch.matmul(query, key) / math.sqrt(kdim // num_heads)
        attn_mask = self.generate_square_subsequent_mask(
            height * width,
            device=attn_score.device,
            dtype=attn_score.dtype,
        )
        attn_score = attn_score + attn_mask
        attn_weights = F.softmax(attn_score, dim=-1)
        attn_weights = F.dropout(attn_weights, p=dropout, training=self.training)
        x = torch.matmul(attn_weights, value)
        x = x.permute(0, 1, 3, 2).contiguous()
        output = x.view(batch_size, out_channels, height, width)

        return output

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),
        dtype: torch.dtype = torch.get_default_dtype(),
    ) -> torch.BoolTensor:
        return _generate_square_subsequent_mask(sz, device=device, dtype=dtype)


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


def _generate_square_subsequent_mask(
    sz: int,
    device: torch.device = torch.device(torch._C._get_default_device()),
    dtype: torch.dtype = torch.get_default_dtype(),
) -> torch.Tensor:
    r"""Generate a square causal mask for the sequence.

    Ported from
    https://github.com/pytorch/pytorch/blob/fdaddec2c3a64f9d0d98b5f71b8eef40a247c0c2/torch/nn/modules/transformer.py#L18-L30

    """
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )
