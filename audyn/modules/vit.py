"""Modules related to Vision Transformer (ViT).
This implementation is typically used for audio spectrogram transformer (AST).
"""

from abc import abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

__all__ = [
    "PositionalPatchEmbedding",
    "PatchEmbedding",
]


class _PatchEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        insert_cls_token: bool = False,
        insert_dist_token: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        super().__init__()

        if insert_dist_token and not insert_cls_token:
            raise ValueError("When insert_dist_token=True, insert_cls_token should be True.")

        if insert_cls_token:
            cls_token = torch.empty(
                (embedding_dim,),
                **factory_kwargs,
            )
            self.cls_token = nn.Parameter(cls_token)
        else:
            self.register_parameter("cls_token", None)

        if insert_dist_token:
            dist_token = torch.empty(
                (embedding_dim,),
                **factory_kwargs,
            )
            self.dist_token = nn.Parameter(dist_token)
        else:
            self.register_parameter("dist_token", None)

    def reset_head_tokens(self) -> None:
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token.data, std=0.02)

        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token.data, std=0.02)

    @property
    def insert_cls_token(self) -> bool:
        return self.cls_token is not None

    @property
    def insert_dist_token(self) -> bool:
        return self.dist_token is not None

    @abstractmethod
    def spectrogram_to_patches(self, input: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to patches."""

    def patches_to_sequence(self, input: Union[torch.Tensor, torch.BoolTensor]) -> torch.Tensor:
        """Convert 3D (batch_size, height, width) or 4D (batch_size, embedding_dim, height, width)
        tensor to shape (batch_size, length, *) for input of Transformer.

        Args:
            input (torch.Tensor): Patches of shape (batch_size, height, width) or
                (batch_size, embedding_dim, height, width).

        Returns:
            torch.Tensor: Sequence of shape (batch_size, length) or
                (batch_size, length, embedding_dim).

        """
        n_dims = input.dim()

        if n_dims == 3:
            batch_size, height, width = input.size()
            output = input.view(batch_size, height * width)
        elif n_dims == 4:
            batch_size, embedding_dim, height, width = input.size()
            x = input.view(batch_size, embedding_dim, height * width)
            output = x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError("Only 3D and 4D tensors are supported.")

        return output

    def split_sequence(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split sequence to head tokens and content tokens.

        Args:
            sequence (torch.Tensor): Sequence containing head tokens, i.e. class and distillation
                tokens. The shape is (batch_size, length, embedding_dim).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Head tokens of shape (batch_size, num_head_tokens, embedding_dim).
                - torch.Tensor: Sequence of shape
                    (batch_size, length - num_head_tokens, embedding_dim).

        .. note::

            This method is applicable even when sequence does not contain head tokens. In that
            case, an empty sequnce is returened as the first item of returned tensors.

        """
        length = sequence.size(-2)
        num_head_tokens = 0

        if self.cls_token is not None:
            num_head_tokens += 1

        if self.dist_token is not None:
            num_head_tokens += 1

        head_tokens, sequence = torch.split(
            sequence, [num_head_tokens, length - num_head_tokens], dim=-2
        )

        return head_tokens, sequence

    @abstractmethod
    def compute_output_shape(self, n_bins: int, n_frames: int) -> Tuple[int, int]:
        """Compute output shape from shape of spectrogram."""


class PositionalPatchEmbedding(_PatchEmbedding):
    """Patch embedding + trainable positional embedding.

    Args:
        embedding_dim (int): Embedding dimension.
        kernel_size (_size_2_t): Kernel size that corresponds to patch.
        stride (_size_2_t): Stride.
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
        insert_cls_token: bool = False,
        insert_dist_token: bool = False,
        dropout: float = 0,
        n_bins: int = None,
        n_frames: int = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__(
            embedding_dim,
            insert_cls_token=insert_cls_token,
            insert_dist_token=insert_dist_token,
            **factory_kwargs,
        )

        if n_bins is None:
            raise ValueError("n_bins is required.")

        if n_frames is None:
            raise ValueError("n_frames is required.")

        kernel_size = _pair(kernel_size)

        if stride is None:
            stride = kernel_size

        stride = _pair(stride)

        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_bins = n_bins
        self.n_frames = n_frames

        self.conv2d = nn.Conv2d(
            1,
            embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
        )

        height, width = self.compute_output_shape(n_bins, n_frames)
        positional_embedding = torch.empty((embedding_dim, height, width), **factory_kwargs)
        self.positional_embedding = nn.Parameter(positional_embedding)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        self.reset_head_tokens()

        # based on official implementation
        nn.init.trunc_normal_(self.positional_embedding.data, std=0.02)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of PositionalPatchEmbedding.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, n_bins, n_frames).

        Returns:
            torch.Tensor: (batch_size, height * width + num_head_tokens, embedding_dim),
                where `num_head_tokens` represents number of tokens for [CLS] and [DIST].

        """
        positional_embedding = self.positional_embedding
        _, n_bins, n_frames = input.size()
        x = input.unsqueeze(dim=-3)
        x = self.conv2d(x)
        x = x + self.resample_positional_embedding(
            positional_embedding,
            n_bins,
            n_frames,
        )
        x = self.patches_to_sequence(x)
        batch_size = x.size(0)

        if self.insert_dist_token:
            dist_token = self.dist_token.expand((batch_size, 1, -1))
            x = torch.cat([dist_token, x], dim=-2)

        if self.insert_cls_token:
            cls_token = self.cls_token.expand((batch_size, 1, -1))
            x = torch.cat([cls_token, x], dim=-2)

        output = self.dropout(x)

        return output

    def spectrogram_to_patches(self, input: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to patches."""
        conv2d = self.conv2d
        batch_size, n_bins, n_frames = input.size()
        x = input.view(batch_size, 1, n_bins, n_frames)
        x = F.unfold(
            x,
            kernel_size=conv2d.kernel_size,
            dilation=conv2d.dilation,
            padding=conv2d.padding,
            stride=conv2d.stride,
        )
        height, width = self.compute_output_shape(n_bins, n_frames)

        output = x.view(batch_size, -1, height, width)

        return output

    def resample_positional_embedding(
        self,
        positional_embedding: Union[torch.Tensor],
        n_bins: int,
        n_frames: int,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        """Resample positional embedding.

        Args:
            positional_embedding (torch.Tensor): Positional embedding of shape
                (embedding_dim, height, width).
            n_bins (int): Number of bins, not height.
            n_frames (int): Number of frames, not width.
            mode (str): Interpolation mode. Default: ``bilinear``.

        Returns:
            torch.Tensor: Resampled positional embedding of shape (embedding_dim, height', width').

        """
        _, height_org, width_org = positional_embedding.size()
        height, width = self.compute_output_shape(n_bins, n_frames)

        if width_org > width:
            start_idx = width_org // 2 - width // 2
            _, positional_embedding, _ = torch.split(
                positional_embedding,
                [start_idx, width, width_org - width - start_idx],
                dim=-1,
            )
        elif width > width_org:
            positional_embedding = positional_embedding.unsqueeze(dim=0)
            positional_embedding = F.interpolate(
                positional_embedding, size=(height_org, width), mode=mode
            )
            positional_embedding = positional_embedding.squeeze(dim=0)

        if height_org > height:
            start_idx = height_org // 2 - height // 2
            _, positional_embedding, _ = torch.split(
                positional_embedding,
                [start_idx, height, height_org - height - start_idx],
                dim=-1,
            )
        elif height > height_org:
            positional_embedding = positional_embedding.unsqueeze(dim=0)
            positional_embedding = F.interpolate(
                positional_embedding, size=(height, width), mode=mode
            )
            positional_embedding = positional_embedding.squeeze(dim=0)

        output = positional_embedding

        return output

    def compute_output_shape(self, n_bins: int, n_frames: int) -> Tuple[int, int]:
        Kh, Kw = self.conv2d.kernel_size
        Sh, Sw = self.conv2d.stride
        height = (n_bins - Kh) // Sh + 1
        width = (n_frames - Kw) // Sw + 1

        return height, width


class PatchEmbedding(_PatchEmbedding):
    """Patch embedding w/o positional embedding.

    Args:
        embedding_dim (int): Embedding dimension.
        kernel_size (_size_2_t): Kernel size that corresponds to patch.
        stride (_size_2_t): Stride.
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
        insert_cls_token: bool = False,
        insert_dist_token: bool = False,
        dropout: float = 0,
        n_bins: int = None,
        n_frames: int = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__(
            embedding_dim,
            insert_cls_token=insert_cls_token,
            insert_dist_token=insert_dist_token,
            **factory_kwargs,
        )

        if n_bins is None:
            raise ValueError("n_bins is required.")

        if n_frames is None:
            raise ValueError("n_frames is required.")

        kernel_size = _pair(kernel_size)

        if stride is None:
            stride = kernel_size

        stride = _pair(stride)

        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_bins = n_bins
        self.n_frames = n_frames

        self.conv2d = nn.Conv2d(
            1,
            embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        self.reset_head_tokens()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of PositionalPatchEmbedding.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, n_bins, n_frames).

        Returns:
            torch.Tensor: (batch_size, height * width + num_head_tokens, embedding_dim),
                where `num_head_tokens` represents number of tokens for [CLS] and [DIST].

        """
        x = input.unsqueeze(dim=-3)
        x = self.conv2d(x)
        x = self.patches_to_sequence(x)
        batch_size = x.size(0)

        if self.insert_dist_token:
            dist_token = self.dist_token.expand((batch_size, 1, -1))
            x = torch.cat([dist_token, x], dim=-2)

        if self.insert_cls_token:
            cls_token = self.cls_token.expand((batch_size, 1, -1))
            x = torch.cat([cls_token, x], dim=-2)

        output = self.dropout(x)

        return output

    def spectrogram_to_patches(self, input: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to patches."""
        conv2d = self.conv2d
        batch_size, n_bins, n_frames = input.size()
        x = input.view(batch_size, 1, n_bins, n_frames)
        x = F.unfold(
            x,
            kernel_size=conv2d.kernel_size,
            dilation=conv2d.dilation,
            padding=conv2d.padding,
            stride=conv2d.stride,
        )
        height, width = self.compute_output_shape(n_bins, n_frames)

        output = x.view(batch_size, -1, height, width)

        return output

    def compute_output_shape(self, n_bins: int, n_frames: int) -> Tuple[int, int]:
        Kh, Kw = self.conv2d.kernel_size
        Sh, Sw = self.conv2d.stride
        height = (n_bins - Kh) // Sh + 1
        width = (n_frames - Kw) // Sw + 1

        return height, width
