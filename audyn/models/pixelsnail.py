"""PixelSNAIL model.

See https://arxiv.org/abs/1712.09763 for the details.
TODO: improve architecture to reproduce exactly
"""

import copy
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t

from ..modules.pixelsnail import CausalConv2d, PixelBlock, _get_activation

__all__ = [
    "PixelSNAIL",
    "DiscretePixelSNAIL",
]


class PixelSNAIL(nn.Module):
    """PixelSNAIL."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: _size_2_t,
        num_heads: int,
        num_blocks: int,
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

        self.num_blocks = num_blocks

        self.conv2d_in = CausalConv2d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
        )

        backbone = []

        for _ in range(num_blocks):
            # in_channels is treated as skip_channels in PixelBlock.
            block = PixelBlock(
                hidden_channels,
                in_channels,
                kernel_size=kernel_size,
                num_heads=num_heads,
                num_repeats=num_repeats,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
                weight_regularization=weight_regularization,
                activation=activation,
                **factory_kwargs,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)
        self.conv2d_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, **factory_kwargs)

        if isinstance(activation, str):
            activation = _get_activation(activation)
        else:
            # NOTE: Activations are not shared with each other.
            activation = copy.deepcopy(activation)

        self.activation = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of PixelSNAIL.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, out_channels, height, width).

        """
        num_blocks = self.num_blocks

        # to ensure causailty, shift features
        x_horozontal = F.pad(input, (0, 0, 1, -1))
        x_vertical = F.pad(input, (1, -1))
        x_skip = x_horozontal + x_vertical

        x = self.conv2d_in(x_skip)

        for block_idx in range(num_blocks):
            x = self.backbone[block_idx](x, x_skip)

        x = self.conv2d_out(x)
        output = self.activation(x)

        return output


class DiscretePixelSNAIL(nn.Module):
    """PixelSNAIL for discrete distribution.

    Args:
        embedding (nn.Embedding or callable): Embedding layer to transform
            discrete input (batch_size, height, width) to continuous
            feature (batch_size, height, width, embedding_dim). Typically,
            ``nn.Embedding`` or ``F.one_hot`` is used.
        backbone (PixelSNAIL): Backbone PixelSNAIL model.
        distribution (str): Name of discrete distribution.

    .. note::

        When ``distribution=categorical``, number of output channels of ``backbone``
        should be equal to number of classes in ``embedding``.

    """

    def __init__(
        self,
        embedding: Union[nn.Embedding, Callable[[torch.LongTensor], torch.Tensor]],
        backbone: PixelSNAIL,
        distribution: str = "categorical",
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.backbone = backbone
        self.distribution = distribution

        assert distribution == "categorical", "Only categorical is supported as distribution."

    def forward(self, input: torch.LongTensor) -> torch.Tensor:
        """Forward pass of DiscretePixelSNAIL.

        Args:
            input (torch.LongTensor): Input feature of shape (batch_size, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, num_classes, height, width).

        """
        x = self.embedding(input)
        x = x.permute(0, 3, 1, 2)
        output = self.backbone(x)

        return output

    @torch.no_grad()
    def inference(
        self,
        initial_state: torch.LongTensor,
        height: int = 1,
        width: int = 1,
    ) -> torch.Tensor:
        """Forward pass of PixelCNN.

        Args:
            initial_state (torch.LongTensor): (batch_size, 1, 1).
            height (int): Height of output feature.
            width (int): Width of output feature.

        Returns:
            torch.Tensor: Output feature of shape (batch_size, height, width).

        """
        distribution = self.distribution

        # remove redundancy
        batch_size = initial_state.size(0)
        output = F.pad(initial_state, (0, width - 1, 0, height - 1))

        for row_idx in range(height):
            for column_idx in range(width):
                x = F.pad(output, (0, 0, 0, -(height - 1 - row_idx)))
                x = self.forward(x)
                last_output = F.pad(x, (-column_idx, -(width - 1 - column_idx), -row_idx, 0))
                last_output = last_output.view(batch_size, -1)

                if distribution == "categorical":
                    # sampling from categorical distribution
                    last_output = torch.softmax(last_output, dim=1)
                    last_output = torch.distributions.Categorical(last_output).sample()
                else:
                    raise NotImplementedError(f"{distribution} is not supported as distribution.")

                output[:, row_idx, column_idx] = last_output

        return output
