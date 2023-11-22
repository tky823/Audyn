import copy
import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

from audyn.modules.pixelcnn import CausalConv2d as PixelCNNCausalConv2d
from audyn.modules.pixelsnail import _generate_square_subsequent_mask, _get_activation

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class PixelBlock(nn.Module):
    """Block of PixelSNAIL.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels used in ``ResidualBlock2d``.
        kernel_size (_size_2_t): Kernel size in convolutions.
        num_heads (int): Number of heads in attention.
        num_repeats (int): Number of repeats of ``ResidualBlock2d``.
        dropout (float): Dropout rate in attention. Default: ``0.0``.
        weight_regularization (str, optional): Weight regularization.
        activation (str, nn.Module, or callable): Activation function. Default: ``elu``.

    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: _size_2_t,
        num_heads: int,
        num_repeats: int,
        dropout: float = 0.1,
        auxiliary_channels: Optional[int] = None,
        conditional_channels: Optional[int] = None,
        weight_regularization: Optional[str] = "weight_norm",
        activation: Optional[
            Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = "elu",
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.num_repeats = num_repeats

        backbone = []

        for _ in range(num_repeats):
            block = ResidualBlock(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                causal=True,
                weight_regularization=weight_regularization,
                activation=activation,
                dropout=dropout,
                auxiliary_channels=auxiliary_channels,
                conditional_channels=conditional_channels,
                **factory_kwargs,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)

        self.q_proj = ResidualBlock(
            in_channels + 2,
            in_channels,
            kernel_size=1,
            causal=False,
            weight_regularization=weight_regularization,
            activation=activation,
            dropout=dropout,
            **factory_kwargs,
        )
        self.k_proj = ResidualBlock(
            2 * in_channels + 2,
            in_channels,
            kernel_size=1,
            causal=False,
            weight_regularization=weight_regularization,
            activation=activation,
            dropout=dropout,
            **factory_kwargs,
        )
        self.mha2d = CausalAttention2d(
            in_channels // 2,
            in_channels + 2,
            2 * in_channels + 2,
            num_heads=num_heads,
            dropout=dropout,
            weight_regularization=weight_regularization,
        )
        self.out_proj = ResidualBlock(
            in_channels,
            in_channels,
            kernel_size=1,
            weight_regularization=weight_regularization,
            auxiliary_channels=in_channels // 2,
            dropout=dropout,
            **factory_kwargs,
        )

    def forward(
        self,
        input: torch.Tensor,
        background: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of ConvBlock2d.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, height, width).
            background (torch.Tensor): Background embedding of shape
                (batch_size, 2, height, width) or (2, height, width).
            conditioning (torch.Tensor, optional): Conditional feature of
                shape (batch_size, conditional_channels, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, in_channels, height, width).

        """
        num_repeats = self.num_repeats

        batch_size, _, height, width = input.size()

        if background.dim() == 3:
            x_background = background.unsqueeze(dim=0)
            x_background = x_background.expand(batch_size, 2, height, width)
        elif x_background.dim() == 4:
            x_background = background
        else:
            raise ValueError("background is expected 3D or 4D.")

        x = input

        for repeat_idx in range(num_repeats):
            x = self.backbone[repeat_idx](x, conditioning=conditioning)

        query = torch.cat([x, x_background], dim=1)
        query = self.q_proj(query)
        key = torch.cat([x, input, x_background], dim=1)
        key = self.k_proj(key)
        attn_output = self.mha2d(query, key)
        output = self.out_proj(x, attn_output)

        return output

    def weight_norm_(self) -> None:
        """Set weight_norm to modules."""
        for block in self.backbone:
            block: ResidualBlock
            block.weight_norm_()

        self.q_proj.weight_norm_()
        self.k_proj.weight_norm_()
        self.mha2d.weight_norm_()
        self.out_proj.weight_norm_()

    def remove_weight_norm_(self) -> None:
        """Remove weight_norm from module."""
        for block in self.backbone:
            block: ResidualBlock
            block.weight_norm_()

        self.q_proj.weight_norm_()
        self.k_proj.weight_norm_()
        self.mha2d.weight_norm_()
        self.out_proj.weight_norm_()


class ResidualBlock(nn.Module):
    """ResidualBlock.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        kernel_size (_size_2_t): Kernel size in convolutions.
        causal: Causality. If ``causal=True``, causal convolution is used.
        weight_regularization (str, optional): Weight regularization.
        activation (str, nn.Module, or callable): Activation function. Default: ``elu``.
        dropout (float): Dropout rate in attention. Default: ``0.0``.

    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: _size_2_t,
        causal: bool = False,
        weight_regularization: Optional[str] = "weight_norm",
        activation: Optional[
            Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = "elu",
        dropout: float = 0.1,
        auxiliary_channels: Optional[int] = None,
        conditional_channels: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        kernel_size = _pair(kernel_size)

        if causal:
            # To be compatible with definition in CausalConv2d,
            # re-define kernel_size.
            kh, kw = kernel_size
            kernel_size = 2 * kh - 1, kw

        self.kernel_size = kernel_size
        self.causal = causal

        if isinstance(activation, str):
            activation_1 = _get_activation(activation)
            activation_2 = _get_activation(activation)

            if auxiliary_channels is None:
                aux_activation = None
            else:
                aux_activation = _get_activation(activation)
        else:
            # NOTE: Activations are not shared with each other.
            activation_1 = copy.deepcopy(activation)
            activation_2 = copy.deepcopy(activation)

            if auxiliary_channels is None:
                aux_activation = None
            else:
                aux_activation = copy.deepcopy(activation)

        self.activation_1 = activation_1

        if causal:
            # NOTE: capture_center=True also ensures causality.
            self.conv2d_1 = PixelCNNCausalConv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                capture_center=False,
                **factory_kwargs,
            )
        else:
            self.conv2d_1 = nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                **factory_kwargs,
            )

        # auxiliary input
        if auxiliary_channels is None:
            self.aux_activation = None
            self.aux_conv2d = None
        else:
            self.aux_activation = aux_activation
            self.aux_conv2d = nn.Conv2d(
                auxiliary_channels,
                hidden_channels,
                kernel_size=1,
                **factory_kwargs,
            )

        self.activation_2 = activation_2
        self.dropout2d = nn.Dropout(dropout)

        if causal:
            # NOTE: capture_center=True also ensures causality.
            self.conv2d_2 = PixelCNNCausalConv2d(
                hidden_channels,
                2 * in_channels,
                kernel_size=kernel_size,
                capture_center=False,
                **factory_kwargs,
            )
        else:
            self.conv2d_2 = nn.Conv2d(
                hidden_channels,
                2 * in_channels,
                kernel_size=kernel_size,
                **factory_kwargs,
            )

        # conditional input
        if conditional_channels is None:
            self.conditional_conv2d = None
        else:
            self.conditional_conv2d = nn.Conv2d(
                conditional_channels,
                2 * in_channels,
                kernel_size=1,
                bias=False,
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

    def forward(
        self,
        input: torch.Tensor,
        auxiliary: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, height, width).
            auxiliary (torch.Tensor, optional): Auxiliary feature of
                shape (batch_size, auxiliary_channels, height, width).
            conditioning (torch.Tensor, optional): Conditional feature of
                shape (batch_size, conditional_channels, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, in_channels, height, width).

        """
        kh, kw = self.kernel_size

        padding_height = (kh - 1) // 2
        padding_width = (kw - 1) // 2
        padding_size = (padding_width, padding_width, padding_height, padding_height)

        x = self.activation_1(input)
        x = F.pad(x, padding_size)
        x = self.conv2d_1(x)

        if auxiliary is not None:
            x_auxiliary = self.aux_activation(auxiliary)
            x = x + self.aux_conv2d(x_auxiliary)

        x = self.activation_2(x)
        x = self.dropout2d(x)
        x = F.pad(x, padding_size)
        x = self.conv2d_2(x)

        if conditioning is not None:
            x = x + self.conditional_conv2d(conditioning)

        x = self.glu(x)
        output = x + input

        return output

    def weight_norm_(self) -> None:
        """Set weight_norm to convolutions."""
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.conv2d_1 = weight_norm_fn(self.conv2d_1)
        self.conv2d_2 = weight_norm_fn(self.conv2d_2)

        if self.aux_conv2d is not None:
            self.aux_conv2d = weight_norm_fn(self.aux_conv2d)

        if self.conditional_conv2d is not None:
            self.conditional_conv2d = weight_norm_fn(self.conditional_conv2d)

    def remove_weight_norm_(self) -> None:
        """Remove weight_norm from convolutions."""
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.conv2d_1 = remove_weight_norm_fn(self.conv2d_1, *remove_weight_norm_args)
        self.conv2d_2 = remove_weight_norm_fn(self.conv2d_2, *remove_weight_norm_args)

        if self.aux_conv2d is not None:
            self.aux_conv2d = remove_weight_norm_fn(self.aux_conv2d)

        if self.conditional_conv2d is not None:
            self.conditional_conv2d = remove_weight_norm_fn(self.conditional_conv2d)

    def spectral_norm_(self) -> None:
        """Set spectral_norm to convolutions."""
        if IS_TORCH_LT_2_1:
            spectral_norm_fn = nn.utils.spectral_norm
        else:
            spectral_norm_fn = nn.utils.parametrizations.spectral_norm

        self.conv2d_1 = spectral_norm_fn(self.conv2d_1)
        self.conv2d_2 = spectral_norm_fn(self.conv2d_2)

        if self.aux_conv2d is not None:
            self.aux_conv2d = spectral_norm_fn(self.aux_conv2d)

        if self.conditional_conv2d is not None:
            self.conditional_conv2d = spectral_norm_fn(self.conditional_conv2d)

    def remove_spectral_norm_(self) -> None:
        """Remove spectral_norm from convolutions."""
        if IS_TORCH_LT_2_1:
            remove_spectral_norm_fn = nn.utils.remove_spectral_norm
            remove_spectral_norm_args = ()
        else:
            remove_spectral_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_spectral_norm_args = ("weight",)

        self.conv2d_1 = remove_spectral_norm_fn(self.conv2d_1, *remove_spectral_norm_args)
        self.conv2d_2 = remove_spectral_norm_fn(self.conv2d_2, *remove_spectral_norm_args)

        if self.aux_conv2d is not None:
            self.aux_conv2d = remove_spectral_norm_fn(self.aux_conv2d)

        if self.conditional_conv2d is not None:
            self.conditional_conv2d = remove_spectral_norm_fn(self.conditional_conv2d)


class CausalAttention2d(nn.Module):
    """Cross attention with causality for 2D input.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Embedding dimension of values, which is equal to number of
            output channels. ``out_channels`` should be divisible by ``num_heads``.
        kdim (int): Embedding dimension of keys. ``kdim`` should be divisible by ``num_heads``.
        num_heads (int): Number of heads in attention.
        dropout (float): Dropout rate in attention. Default: ``0.0``.
        weight_regularization (str, optional): Weight regularization.

    """

    def __init__(
        self,
        embed_dim: int,
        qdim: int,
        kdim: int,
        num_heads: int,
        dropout: float = 0.0,
        weight_regularization: Optional[str] = "weight_norm",
    ) -> None:
        super().__init__()

        self.q_proj = nn.Linear(qdim, embed_dim)
        self.k_proj = nn.Linear(kdim, embed_dim)
        self.v_proj = nn.Linear(kdim, embed_dim)

        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) should be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.qdim, self.kdim = qdim, kdim
        self.num_heads = num_heads
        self.dropout = dropout

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

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Forward pass of CausalSelfAttention2d.

        Args:
            query (torch.Tensor): Query of shape (batch_size, qdim, height, width).
            key (torch.Tensor): Key of shape (batch_size, kdim, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, embed_dim, height, width).

        """
        embed_dim = self.embed_dim
        qdim, kdim = self.qdim, self.kdim
        num_heads = self.num_heads
        dropout = self.dropout
        batch_size, _, height, width = query.size()

        query = query.permute(0, 2, 3, 1).contiguous()
        query = query.view(batch_size, height * width, qdim)
        key = key.permute(0, 2, 3, 1).contiguous()
        key = key.view(batch_size, height * width, kdim)

        query = self.q_proj(query)
        value = self.v_proj(key)
        key = self.k_proj(key)
        query = query.view(batch_size, height * width, num_heads, embed_dim // num_heads)
        key = key.view(batch_size, height * width, num_heads, embed_dim // num_heads)
        value = value.view(batch_size, height * width, num_heads, embed_dim // num_heads)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attn_score = torch.matmul(query, key) / math.sqrt(embed_dim // num_heads)
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
        output = x.view(batch_size, embed_dim, height, width)

        return output

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),
        dtype: torch.dtype = torch.get_default_dtype(),
    ) -> torch.BoolTensor:
        return _generate_square_subsequent_mask(sz, device=device, dtype=dtype)

    def weight_norm_(self) -> None:
        """Set weight_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.q_proj = weight_norm_fn(self.q_proj)
        self.k_proj = weight_norm_fn(self.k_proj)
        self.v_proj = weight_norm_fn(self.v_proj)

    def remove_weight_norm_(self) -> None:
        """Remove weight_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.q_proj = remove_weight_norm_fn(self.q_proj, *remove_weight_norm_args)
        self.k_proj = remove_weight_norm_fn(self.k_proj, *remove_weight_norm_args)
        self.v_proj = remove_weight_norm_fn(self.v_proj, *remove_weight_norm_args)

    def spectral_norm_(self) -> None:
        """Set spectral_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            spectral_norm_fn = nn.utils.spectral_norm
        else:
            spectral_norm_fn = nn.utils.parametrizations.spectral_norm

        self.q_proj = spectral_norm_fn(self.q_proj)
        self.k_proj = spectral_norm_fn(self.k_proj)
        self.v_proj = spectral_norm_fn(self.v_proj)

    def remove_spectral_norm_(self) -> None:
        """Remove spectral_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            remove_spectral_norm_fn = nn.utils.remove_spectral_norm
            remove_spectral_norm_args = ()
        else:
            remove_spectral_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_spectral_norm_args = ("weight",)

        self.q_proj = remove_spectral_norm_fn(self.q_proj, *remove_spectral_norm_args)
        self.k_proj = remove_spectral_norm_fn(self.k_proj, *remove_spectral_norm_args)
        self.v_proj = remove_spectral_norm_fn(self.v_proj, *remove_spectral_norm_args)
