import warnings
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.tasnet import _Decoder, _Encoder
from ..modules.dprnn import DPRNN
from ..modules.tasnet import get_layer_norm, get_nonlinear
from ..modules.transforms import OverlapAdd1d, Segment1d
from .tasnet import TasNet

__all__ = [
    "DPRNNTasNet",
    "Separator",
    "DPRNNTasNetSeparator",
]


class DPRNNTasNet(TasNet):
    """Dual-path RNN TasNet."""

    def __init__(
        self,
        encoder: _Encoder,
        decoder: _Decoder,
        separator: "Separator",
        num_sources: int = None,
    ) -> None:
        if not isinstance(separator, Separator):
            warnings.warn(
                f"Unexpected separator {type(separator)} is given.",
                UserWarning,
                stacklevel=2,
            )

        super().__init__(
            encoder,
            decoder,
            separator,
            num_sources=num_sources,
        )

    @classmethod
    def build_from_config(
        cls,
        encoder: _Encoder,
        decoder: _Decoder,
        bottleneck_channels: int = 64,
        hidden_channels: int = 128,
        chunk_size: int = 100,
        hop_size: int = 50,
        num_blocks: int = 6,
        norm: Optional[Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = True,
        mask_nonlinear: str = "sigmoid",
        is_causal: bool = False,
        rnn: Union[str, nn.Module, Callable[[torch.Tensor], Tuple[torch.Tensor, Any]]] = "lstm",
        num_sources: int = 2,
        eps: float = 1e-8,
    ) -> "DPRNNTasNet":
        num_basis = encoder.num_basis

        assert decoder.num_basis == num_basis

        separator = Separator(
            num_basis,
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            chunk_size=chunk_size,
            hop_size=hop_size,
            num_blocks=num_blocks,
            norm=norm,
            mask_nonlinear=mask_nonlinear,
            is_causal=is_causal,
            rnn=rnn,
            num_sources=num_sources,
            eps=eps,
        )

        return cls(
            encoder,
            decoder,
            separator,
            num_sources=num_sources,
        )


class Separator(nn.Module):
    """Separator for DPRNN-TasNet."""

    def __init__(
        self,
        num_features: int,
        bottleneck_channels: int = 64,
        hidden_channels: int = 128,
        chunk_size: int = 100,
        hop_size: int = 50,
        num_blocks: int = 6,
        norm: Optional[Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = True,
        mask_nonlinear: str = "sigmoid",
        is_causal: bool = False,
        rnn: Union[str, nn.Module, Callable[[torch.Tensor], Tuple[torch.Tensor, Any]]] = "lstm",
        num_sources: int = 2,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.num_sources = num_sources
        self.chunk_size = chunk_size
        self.hop_size = hop_size

        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = get_layer_norm(norm, num_features, eps=eps)
        elif isinstance(norm, bool):
            if norm:
                norm = "cLN" if is_causal else "gLN"
                self.norm = get_layer_norm(norm, num_features, is_causal=is_causal, eps=eps)
            else:
                self.norm = None
        else:
            raise ValueError(f"{type(norm)} is not supported as norm.")

        self.bottleneck_conv1d = nn.Conv1d(
            num_features, bottleneck_channels, kernel_size=1, stride=1
        )

        self.segment1d = Segment1d(chunk_size, hop_size)
        self.dprnn = DPRNN(
            bottleneck_channels,
            hidden_channels,
            num_blocks=num_blocks,
            is_causal=is_causal,
            norm=norm,
            rnn=rnn,
            eps=eps,
        )
        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)

        self.prelu = nn.PReLU()
        self.mask_conv1d = nn.Conv1d(
            bottleneck_channels, num_sources * num_features, kernel_size=1, stride=1
        )

        if mask_nonlinear == "sigmoid":
            kwargs = {}
        elif mask_nonlinear == "softmax":
            kwargs = {"dim": 1}
        else:
            kwargs = {}

        self.mask_nonlinear = get_nonlinear(mask_nonlinear, nonlinear_kwargs=kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of Separator used in DPRNN-TasNet.

        Args:
            input (torch.Tensor): Mixture of shape (batch_size, num_features, num_frames).

        Returns:
            torch.Tensor: Separated feature of shape
                (batch_size, num_sources, num_features, num_frames).

        """
        num_features, num_sources = self.num_features, self.num_sources
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, num_features, num_frames = input.size()

        padding = (hop_size - (num_frames - chunk_size) % hop_size) % hop_size
        padding_left = padding // 2
        padding_right = padding - padding_left

        x = self.norm(input)
        x = self.bottleneck_conv1d(x)
        x = F.pad(x, (padding_left, padding_right))
        x = self.segment1d(x)
        x = self.dprnn(x)
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x)
        x = self.mask_conv1d(x)
        x = self.mask_nonlinear(x)
        output = x.view(batch_size, num_sources, num_features, num_frames)

        return output


class DPRNNTasNetSeparator(Separator):
    """Alias of Separator."""
