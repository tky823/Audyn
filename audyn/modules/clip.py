from typing import Optional

import torch
import torch.nn as nn
from packaging import version
from torch.nn.common_types import _size_2_t

from .vit import PositionalPatchEmbedding as _PositionalPatchEmbedding

__all__ = [
    "PositionalPatchEmbedding",
    "GELU",
]

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class PositionalPatchEmbedding(_PositionalPatchEmbedding):
    """Patch embedding + trainable positional embedding for CLIP.

    Args:
        in_channels (int): Number of input channels.
        embedding_dim (int): Embedding dimension.
        kernel_size (_size_2_t): Kernel size that corresponds to patch.
        stride (_size_2_t): Stride.
        bias (bool): If ``True``, bias is added to embedding.
        insert_cls_token (bool): If ``True``, class token is inserted to beginning of sequence.
        insert_dist_token (bool): If ``True``, distillation token is inserd to beginning sequence.
        dropout (float): Dropout rate.
        n_bins (int): Number of input bins.
        n_frames (int): Number of input frames.

    .. note::

        Unlike official implementation (of AST), trainable positional embedding for CLS (and DIST)
        token(s) are omitted in terms of redundancy.

    """

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        bias: bool = True,
        insert_cls_token: bool = False,
        insert_dist_token: bool = False,
        dropout: float = 0,
        n_bins: int = None,
        n_frames: int = None,
        layer_norm_eps: float = 1e-5,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__(
            in_channels,
            embedding_dim,
            kernel_size,
            stride=stride,
            bias=bias,
            insert_cls_token=insert_cls_token,
            insert_dist_token=insert_dist_token,
            dropout=dropout,
            n_bins=n_bins,
            n_frames=n_frames,
            **factory_kwargs,
        )

        self.norm = nn.LayerNorm(
            embedding_dim,
            eps=layer_norm_eps,
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = super().forward(input)
        output = self.norm(x)

        return output


class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input * torch.sigmoid(1.702 * input)

        return output
