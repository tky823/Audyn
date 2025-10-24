from typing import Optional, Union

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
        num_chunks (int, optional): Number of chunks. This parameters is useful to support
            multiple chunks.
        fusion (bool or nn.Module, optional): Whether to apply fusion.
        insert_cls_token (bool): If ``True``, class token is inserted to beginning of sequence.
        insert_dist_token (bool): If ``True``, distillation token is inserted to beginning
            sequence.
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
        num_chunks: Optional[int] = None,
        fusion: Optional[Union[bool, nn.Module]] = None,
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

        if fusion is None:
            if num_chunks is None:
                num_chunks = 1
        elif isinstance(fusion, nn.Module):
            if num_chunks is None:
                num_chunks = 4
        elif fusion:
            if num_chunks is None:
                num_chunks = 4

            fusion = FusionBlock(out_channels, out_channels // 4, **factory_kwargs)
        else:
            fusion = None

            if num_chunks is None:
                num_chunks = 1

        if fusion is None:
            local_conv2d = None
        else:
            kernel_height, kernel_width = kernel_size
            stride_height, stride_width = stride

            local_conv2d = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_height, 3 * kernel_width),
                stride=(stride_height, 3 * stride_width),
                **factory_kwargs,
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_chunks = num_chunks
        self.n_bins = n_bins
        self.height = height
        self.width = width

        self.norm1 = nn.BatchNorm2d(n_bins, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, **factory_kwargs
        )
        self.local_conv2d = local_conv2d
        self.fusion = fusion
        self.norm2 = nn.LayerNorm(out_channels, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        self.reset_head_tokens()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of PositionalPatchEmbedding.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, in_channels, n_bins, n_frames),
                if ``self.fusion`` is ``None``. Otherwise,
                (batch_size, num_chunks, in_channels, n_bins, n_frames). Here, in_channels
                corresponds to number of fused chunks, and num_chunks corresponds to
                number of fused chunks.

        Returns:
            torch.Tensor: (batch_size, height * width + num_head_tokens, embedding_dim),
                where `num_head_tokens` represents number of tokens for [CLS] and [DIST].

        """
        n_dims = input.dim()

        if self.fusion is None:
            if n_dims == 3:
                x = input.unsqueeze(dim=-3)
            elif n_dims == 4:
                x = input
            else:
                raise ValueError(
                    f"Only 3D or 4D input is supported, but {n_dims}D input is given."
                )
        else:
            if n_dims == 4:
                x = input.unsqueeze(dim=-3)
            elif n_dims == 5:
                x = input
            else:
                raise ValueError(
                    f"Only 4D or 5D input is supported, but {n_dims}D input is given."
                )

        x = self.compute_patch_embedding(x)
        x = self.patches_to_sequence(x)
        x = self.norm2(x)
        x = self.prepend_head_tokens(x)
        output = self.dropout(x)

        return output

    def compute_patch_embedding(self, input: torch.Tensor) -> torch.Tensor:
        """Compute patch embeddings of input feature.

        Args:
            input (torch.Tensor): Spectrogram-like feature of shape
                (batch_size, in_channels, n_bins, n_frames) or
                (batch_size, num_chunks, in_channels, n_bins, n_frames).

        Returns:
            torch.Tensor: Embedded features of shape (batch_size, embedding_dim, height, width).

        """
        n_bins = self.n_bins
        n_frames = self.n_frames
        height = self.height
        width = self.width

        batch_size, *in_channels, _n_bins, _n_frames = input.size()
        x = input.view(batch_size, -1, _n_bins, _n_frames)
        x = x.permute(0, 2, 1, 3)
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

        x = x.view(batch_size, -1, n_bins, (height // n_bins), width)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, -1, height, width)

        if self.fusion is None:
            output = self.conv2d(x)
        else:
            num_chunks, in_channels = in_channels
            x = x.view(batch_size, num_chunks, in_channels, height, width)
            x_global, x_local = torch.split(x, [1, num_chunks - 1], dim=-4)

            # global feature
            x_global = x_global.squeeze(dim=-4)
            x_global = self.conv2d(x_global)
            width_global = x_global.size(-1)

            # local feature
            x_local = x_local.contiguous()
            x_local = x_local.view(batch_size * (num_chunks - 1), in_channels, height, width)
            x_local = self.local_conv2d(x_local)

            *_, out_channels, height_local, width_local = x_local.size()
            x_local = x_local.view(
                batch_size, num_chunks - 1, out_channels, height_local, width_local
            )
            x_local = x_local.permute(0, 2, 3, 1, 4).contiguous()
            x_local = x_local.view(
                batch_size, out_channels, height_local, (num_chunks - 1) * width_local
            )
            width_local = x_local.size(-1)
            x_local = F.pad(x_local, (0, width_global - width_local), "constant", 0)
            output = self.fusion(x_global, x_local)

        return output

    @property
    def n_frames(self) -> int:
        return (self.height * self.width) // self.n_bins


class FusionBlock(nn.Module):
    """Fusion block of CLAP."""

    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        self.local_attn = LocalAttentionBlock(
            num_features,
            hidden_channels,
            **factory_kwargs,
        )
        self.global_attn = GlobalAttentionBlock(
            num_features,
            hidden_channels,
            **factory_kwargs,
        )
        self.gate = nn.Sigmoid()

    def forward(self, input: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = input + residual
        x_local = self.local_attn(x)
        x_global = self.global_attn(x)
        x_gated = self.gate(x_local + x_global)
        output = 2 * input * x_gated + 2 * residual * (1 - x_gated)

        return output


class LocalAttentionBlock(nn.Module):
    """Module to capture local features in fusion block of CLAP."""

    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        self.conv2d1 = nn.Conv2d(
            num_features,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            **factory_kwargs,
        )
        self.norm2d1 = nn.BatchNorm2d(
            hidden_channels,
            **factory_kwargs,
        )
        self.relu2d = nn.ReLU()
        self.conv2d2 = nn.Conv2d(
            hidden_channels,
            num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            **factory_kwargs,
        )
        self.norm2d2 = nn.BatchNorm2d(
            num_features,
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv2d1(input)
        x = self.norm2d1(x)
        x = self.relu2d(x)
        x = self.conv2d2(x)
        output = self.norm2d2(x)

        return output


class GlobalAttentionBlock(nn.Module):
    """Module to capture global features in fusion block of CLAP."""

    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.conv2d1 = nn.Conv2d(
            num_features,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            **factory_kwargs,
        )
        self.norm2d1 = nn.BatchNorm2d(
            hidden_channels,
            **factory_kwargs,
        )
        self.relu2d = nn.ReLU()
        self.conv2d2 = nn.Conv2d(
            hidden_channels,
            num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            **factory_kwargs,
        )
        self.norm2d2 = nn.BatchNorm2d(
            num_features,
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.pool2d(input)
        x = self.conv2d1(x)
        x = self.norm2d1(x)
        x = self.relu2d(x)
        x = self.conv2d2(x)
        output = self.norm2d2(x)

        return output
