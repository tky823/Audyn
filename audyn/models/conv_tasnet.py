import warnings

import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t

from ..models.tasnet import _Decoder, _Encoder
from ..modules.conv_tasnet import TimeDilatedConvNet
from ..modules.tasnet import get_layer_norm, get_nonlinear
from .tasnet import TasNet

__all__ = [
    "ConvTasNet",
    "Separator",
    "ConvTasNetSeparator",
]


class ConvTasNet(TasNet):
    """Conv-TasNet."""

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
        bottleneck_channels: int = 128,
        hidden_channels: int = 256,
        skip_channels: int = 128,
        kernel_size: _size_1_t = 3,
        num_blocks: int = 3,
        num_layers: int = 8,
        dilated: bool = True,
        separable: bool = True,
        nonlinear: str = "prelu",
        norm: bool = True,
        mask_nonlinear: str = "sigmoid",
        is_causal: bool = True,
        num_sources: int = 2,
        eps: float = 1e-8,
    ) -> "ConvTasNet":
        num_basis = encoder.num_basis

        assert decoder.num_basis == num_basis

        separator = Separator(
            num_basis,
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            skip_channels=skip_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            num_layers=num_layers,
            dilated=dilated,
            separable=separable,
            is_causal=is_causal,
            nonlinear=nonlinear,
            norm=norm,
            mask_nonlinear=mask_nonlinear,
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
    """Separator for Conv-TasNet."""

    def __init__(
        self,
        num_features: int,
        bottleneck_channels: int = 128,
        hidden_channels: int = 256,
        skip_channels: int = 128,
        kernel_size: _size_1_t = 3,
        num_blocks: int = 3,
        num_layers: int = 8,
        dilated: bool = True,
        separable: bool = True,
        is_causal: bool = True,
        nonlinear: str = "prelu",
        norm: bool = True,
        mask_nonlinear: str = "sigmoid",
        num_sources: int = 2,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.num_sources = num_sources

        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            norm_type = norm
            self.norm = get_layer_norm(norm_type, num_features, eps=eps)
        elif isinstance(norm, bool):
            if norm:
                norm_type = "cLN" if is_causal else "gLN"
                self.norm = get_layer_norm(norm_type, num_features, is_causal=is_causal, eps=eps)
            else:
                self.norm = None
        else:
            raise ValueError(f"{type(norm)} is not supported as norm.")

        self.bottleneck_conv1d = nn.Conv1d(
            num_features, bottleneck_channels, kernel_size=1, stride=1
        )
        self.tdcn = TimeDilatedConvNet(
            bottleneck_channels,
            hidden_channels=hidden_channels,
            skip_channels=skip_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            num_layers=num_layers,
            dilated=dilated,
            separable=separable,
            is_causal=is_causal,
            nonlinear=nonlinear,
            norm=norm,
        )
        self.prelu = nn.PReLU()
        self.mask_conv1d = nn.Conv1d(
            skip_channels, num_sources * num_features, kernel_size=1, stride=1
        )

        if mask_nonlinear == "sigmoid":
            self.mask_nonlinear = nn.Sigmoid()
        elif mask_nonlinear == "softmax":
            self.mask_nonlinear = nn.Softmax(dim=1)
        else:
            raise ValueError("{} is not supported.".format(mask_nonlinear))

        if mask_nonlinear == "sigmoid":
            kwargs = {}
        elif mask_nonlinear == "softmax":
            kwargs = {"dim": 1}
        else:
            kwargs = {}

        self.mask_nonlinear = get_nonlinear(mask_nonlinear, nonlinear_kwargs=kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of Separator used in Conv-TasNet.

        Args:
            input (torch.Tensor): Mixture of shape (batch_size, num_features, num_frames).

        Returns:
            torch.Tensor: Separated feature of shape
                (batch_size, num_sources, num_features, num_frames).

        """
        num_features = self.num_features
        num_sources = self.num_sources

        batch_size, _, num_frames = input.size()

        x = self.norm(input)
        x = self.bottleneck_conv1d(x)
        x = self.tdcn(x)
        x = self.prelu(x)
        x = self.mask_conv1d(x)
        x = self.mask_nonlinear(x)
        output = x.view(batch_size, num_sources, num_features, num_frames)

        return output


class ConvTasNetSeparator(Separator):
    """Alias of Separator."""
