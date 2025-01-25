from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

from .vit import _PatchEmbedding

__all__ = ["PatchEmbedding"]


class PatchEmbedding(_PatchEmbedding):
    """Patch embedding w/o positional embedding.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels identical to embedding dimension.
        kernel_size (_size_2_t): Kernel size that corresponds to patch.
        stride (_size_2_t): Stride.
        fusion (bool): Whether to apply fusion.
        insert_cls_token (bool): If ``True``, class token is inserted to beginning of sequence.
        insert_dist_token (bool): If ``True``, distillation token is inserd to beginning sequence.
        dropout (float): Dropout rate.
        n_bins (int): Number of input bins.
        height (int): Height of image-like patchgram.
        width (int): Width of image-like patchgram.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        fusion: bool = False,
        insert_cls_token: bool = False,
        insert_dist_token: bool = False,
        dropout: float = 0,
        n_bins: int = None,
        height: int = 256,
        width: int = 256,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__(
            out_channels,
            insert_cls_token=insert_cls_token,
            insert_dist_token=insert_dist_token,
            **factory_kwargs,
        )

        if n_bins is None:
            raise ValueError("n_bins is required.")

        kernel_size = _pair(kernel_size)

        if stride is None:
            stride = kernel_size

        stride = _pair(stride)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.fusion = fusion
        self.n_bins = n_bins
        self.height = height
        self.width = width

        self.norm1 = nn.BatchNorm2d(n_bins)
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        self.reset_head_tokens()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of PositionalPatchEmbedding.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, in_channels, n_bins, n_frames),
                where in_channels corresponds to number of fused chunks.

        Returns:
            torch.Tensor: (batch_size, height * width + num_head_tokens, embedding_dim),
                where `num_head_tokens` represents number of tokens for [CLS] and [DIST].

        """
        x = self.compute_patch_embedding(input)
        x = self.patches_to_sequence(x)
        x = self.norm2(x)
        x = self.prepend_head_tokens(x)
        output = self.dropout(x)

        return output

    def compute_patch_embedding(self, input: torch.Tensor) -> torch.Tensor:
        """Compute patch embeddings of input feature.

        Args:
            input (torch.Tensor): Spectrogram-like feature of shape
                (batch_size, in_channels, n_bins, n_frames).

        Returns:
            torch.Tensor: Embedded features of shape (batch_size, embedding_dim, height, width).

        """
        n_bins = self.n_bins
        n_frames = self.n_frames
        height = self.height
        width = self.width

        batch_size, in_channels, _n_bins, _n_frames = input.size()

        x = input.permute(0, 2, 1, 3)
        x = self.norm1(x)
        x = x.permute(0, 2, 1, 3)

        if _n_frames < n_frames:
            x = F.interpolate(
                x,
                (_n_bins, n_frames),
                mode="bicubic",
                align_corners=True,
            )
        elif _n_frames > n_frames:
            raise ValueError(
                f"Number of temporal frames {_n_frames} should be less than "
                f"or equal to {n_frames}."
            )

        assert x.size(-1) == n_frames

        if _n_bins < n_bins:
            x = F.interpolate(
                x,
                (n_bins, n_frames),
                mode="bicubic",
                align_corners=True,
            )
        elif _n_bins > n_bins:
            raise ValueError(
                f"Number of frequency bins {_n_bins} should be less than or equal to {n_bins}."
            )

        assert x.size(-2) == n_bins

        x = x.view(batch_size, in_channels, n_bins, (height // n_bins), width)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, in_channels, height, width)

        if self.fusion:
            # TODO: implement here
            raise NotImplementedError("fusion=True is not supported.")
        else:
            output = self.conv2d(x)

        return output

    @property
    def n_frames(self) -> int:
        return (self.height * self.width) // self.n_bins
