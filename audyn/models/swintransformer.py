from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t

from ..modules.swintransformer import SwinTransformerEncoderBlock

__all__ = [
    "SwinTransformerEncoder",
]


class SwinTransformerEncoder(nn.Module):
    """SwinTransformerEncoder.

    Args:
        num_blocks (int): Number of ``SwinTransformerEncoderBlock``.
        num_layers (int or list): Number of ``SwinTransformerEncoderLayer`` in
            ``SwinTransformerEncoderBlock``.

    """

    def __init__(
        self,
        d_model: int,
        nhead: Union[int, List[int]],
        dim_feedforward: Union[int, List[int]] = 2048,
        num_blocks: int = 4,
        num_layers: Union[int, List[int]] = 2,
        dropout: float = 0.1,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        height: int = 64,
        width: int = 64,
        window_size: List[_size_2_t] = None,
        share_heads: bool = True,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        norm: Optional[nn.Module] = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        super().__init__()

        if isinstance(nhead, int):
            nhead = [nhead] * num_blocks
        else:
            assert len(nhead) == num_blocks

        if isinstance(dim_feedforward, int):
            dim_feedforward = [dim_feedforward] * num_blocks
        else:
            assert len(dim_feedforward) == num_blocks

        if isinstance(num_layers, int):
            num_layers = [num_layers] * num_blocks
        else:
            assert len(num_layers) == num_blocks

        assert len(window_size) == num_blocks

        backbone = []

        _out_channels = d_model

        for block_idx in range(num_blocks):
            _nhead = nhead[block_idx]
            _hidden_channels = dim_feedforward[block_idx]
            _num_layers = num_layers[block_idx]
            _height = height // (2**block_idx)
            _width = width // (2**block_idx)
            _window_size = window_size[block_idx]

            _in_channels = _out_channels

            if block_idx == num_blocks - 1:
                _out_channels = None
            else:
                _out_channels = 2 * _in_channels

            block = SwinTransformerEncoderBlock(
                _in_channels,
                _out_channels,
                _nhead,
                dim_feedforward=_hidden_channels,
                num_layers=_num_layers,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                height=_height,
                width=_width,
                window_size=_window_size,
                share_heads=share_heads,
                batch_first=batch_first,
                norm_first=norm_first,
                bias=bias,
                **factory_kwargs,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)
        self.norm = norm

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass of SwinTransformerEncoder.

        Args:
            src (torch.Tensor): Sequence of shape (batch_size, height * width, d_model)
                if ``batch_first=True``. Otherwise, (height * width, batch_size, d_model).

        Returns:
            torch.Tensor: Sequence of as same shape as ``src``.

        """
        x = src

        for block in self.backbone:
            x = block(x)

        if self.norm is None:
            output = x
        else:
            output = self.norm(x)

        return output
