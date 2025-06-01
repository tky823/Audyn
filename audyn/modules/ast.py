from typing import Optional

import torch
from torch.nn.common_types import _size_2_t

from .vit import PatchEmbedding as _PatchEmbedding
from .vit import PositionalPatchEmbedding as _PositionalPatchEmbedding


class PositionalPatchEmbedding(_PositionalPatchEmbedding):
    """Patch embedding + trainable positional embedding for Audio Spectrogram Transformer (AST).

    Args:
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
        embedding_dim: int,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        bias: bool = True,
        insert_cls_token: bool = False,
        insert_dist_token: bool = False,
        dropout: float = 0,
        n_bins: int = None,
        n_frames: int = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        in_channels = 1
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of PositionalPatchEmbedding.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, n_bins, n_frames).

        Returns:
            torch.Tensor: (batch_size, height * width + num_head_tokens, embedding_dim),
                where `num_head_tokens` represents number of tokens for [CLS] and [DIST].

        """
        x = input.unsqueeze(dim=-3)
        output = super().forward(x)

        return output


class PatchEmbedding(_PatchEmbedding):
    """Patch embedding w/o positional embedding for Audio Spectrogram Transformer (AST).

    Args:
        embedding_dim (int): Embedding dimension.
        kernel_size (_size_2_t): Kernel size that corresponds to patch.
        stride (_size_2_t): Stride.
        bias (bool): If ``True``, bias is added to embedding.
        insert_cls_token (bool): If ``True``, class token is inserted to beginning of sequence.
        insert_dist_token (bool): If ``True``, distillation token is inserd to beginning sequence.
        dropout (float): Dropout rate.
        n_bins (int): Number of input bins.
        n_frames (int): Number of input frames.

    """

    def __init__(
        self,
        embedding_dim: int,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        bias: bool = True,
        insert_cls_token: bool = False,
        insert_dist_token: bool = False,
        dropout: float = 0,
        n_bins: int = None,
        n_frames: int = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        in_channels = 1
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of PositionalPatchEmbedding.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, n_bins, n_frames).

        Returns:
            torch.Tensor: (batch_size, height * width + num_head_tokens, embedding_dim),
                where `num_head_tokens` represents number of tokens for [CLS] and [DIST].

        """
        x = input.unsqueeze(dim=-3)
        output = super().forward(x)

        return output
