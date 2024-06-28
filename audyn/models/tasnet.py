from abc import abstractmethod
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single

__all__ = [
    "TasNet",
    "Encoder",
    "Decoder",
]


class TasNet(nn.Module):
    """Base class of TasNet.

    Args:
        encoder (_Encoder): Encoder module to transform waveform into latent feature.
        decoder (_Decoder): Decoder module to transform latent feature into waveform.
        separator (nn.Module): Separator module to estimate masks to each source.

    """

    def __init__(
        self,
        encoder: "_Encoder",
        decoder: "_Decoder",
        separator: nn.Module,
        num_sources: int = None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.separator = separator
        self.decoder = decoder

        if num_sources is None:
            raise ValueError("Set num_sources.")

        self.num_sources = num_sources

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Separate input waveform into individual sources.

        Args:
            input (torch.Tensor): Mixture waveform of shape (batch_size, timesteps)
                or (batch_size, in_channels, timesteps).
        Returns:
            torch.Tensor: Separated sources of shape (batch_size, num_sources, timesteps) or
                (batch_size, num_sources, in_channels, timesteps).

        """
        output, _ = self.extract_latent(input)

        return output

    def extract_latent(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract latent feature of each source.

        Args:
            input (torch.Tensor): Mixture waveform of shape (batch_size, timesteps)
                or (batch_size, in_channels, timesteps).
        Returns:
            tuple: Tuple of tensors containing:

                - torch.Tensor: Separated sources of shape (batch_size, num_sources, timesteps) or
                    (batch_size, num_sources, in_channels, timesteps).
                - torch.Tensor: Latent features of shape
                    (batch_size, num_sources, num_basis, num_frames) or
                    (batch_size, num_sources, in_channels, num_basis, num_frames).

        """
        num_sources = self.num_sources
        num_basis = self.num_basis
        (kernel_size,) = _single(self.kernel_size)
        (stride,) = _single(self.stride)

        n_dims = input.dim()

        if n_dims == 2:
            batch_size, timesteps = input.size()
            in_channels = 1
            input = input.unsqueeze(dim=1)
        elif n_dims == 3:
            batch_size, in_channels, timesteps = input.size()
        else:
            raise ValueError(f"{n_dims}D waveform is not supported.")

        padding = (stride - (timesteps - kernel_size) % stride) % stride
        padding_left = padding // 2
        padding_right = padding - padding_left

        input = F.pad(input, (padding_left, padding_right))
        w = self.encoder(input)

        mask = self.separator(w)
        w = w.unsqueeze(dim=1)
        w_hat = w * mask

        latent = w_hat
        w_hat = w_hat.view(batch_size * num_sources, num_basis, -1)
        x_hat = self.decoder(w_hat)

        x_hat = x_hat.view(batch_size, num_sources, in_channels, -1)

        if n_dims == 2:
            x_hat = x_hat.squeeze(dim=-2)

        output = F.pad(x_hat, (-padding_left, -padding_right))

        return output, latent

    @property
    def kernel_size(self) -> _size_1_t:
        return self.encoder.kernel_size

    @property
    def stride(self) -> _size_1_t:
        return self.encoder.stride

    @property
    def num_basis(self) -> int:
        return self.encoder.num_basis


class _Encoder(nn.Module):
    """Base class of encoder for TasNet."""

    @property
    @abstractmethod
    def kernel_size(self) -> _size_1_t:
        pass

    @property
    @abstractmethod
    def stride(self) -> _size_1_t:
        pass

    @property
    @abstractmethod
    def basis(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def num_basis(self) -> int:
        pass


class _Decoder(nn.Module):
    """Base class of decoder for TasNet."""

    @property
    @abstractmethod
    def kernel_size(self) -> _size_1_t:
        pass

    @property
    @abstractmethod
    def stride(self) -> _size_1_t:
        pass

    @property
    @abstractmethod
    def basis(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def num_basis(self) -> int:
        pass


class Encoder(_Encoder):
    """Encoder for TasNet."""

    def __init__(
        self,
        in_channels: int,
        num_basis: int,
        kernel_size: int = 16,
        stride: int = 8,
        nonlinear: Optional[Union[Callable[[torch.Tensor], torch.Tensor], str]] = None,
    ) -> None:
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels,
            num_basis,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        if nonlinear is not None:
            if nonlinear == "relu":
                self.nonlinear = nn.ReLU()
            else:
                raise NotImplementedError("{} is not supported.".format(nonlinear))
        else:
            self.nonlinear = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(input)

        if self.nonlinear is not None:
            output = self.nonlinear(x)
        else:
            output = x

        return output

    @property
    def kernel_size(self) -> _size_1_t:
        return self.conv1d.weight.size(2)

    @property
    def stride(self) -> _size_1_t:
        return self.conv1d.stride

    @property
    def basis(self) -> torch.Tensor:
        return self.conv1d.weight

    @property
    def num_basis(self) -> torch.Tensor:
        return self.conv1d.weight.size(0)


class Decoder(_Decoder):
    """Decoder for TasNet."""

    def __init__(
        self,
        out_channels: int,
        num_basis: int,
        kernel_size: int = 16,
        stride: int = 8,
    ) -> None:
        super().__init__()

        self.conv_transpose1d = nn.ConvTranspose1d(
            num_basis, out_channels, kernel_size=kernel_size, stride=stride, bias=False
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv_transpose1d(input)

        return output

    @property
    def kernel_size(self) -> _size_1_t:
        return self.conv_transpose1d.weight.size(2)

    @property
    def stride(self) -> _size_1_t:
        return self.conv_transpose1d.stride

    @property
    def basis(self) -> torch.Tensor:
        return self.conv_transpose1d.weight

    @property
    def num_basis(self) -> int:
        return self.conv_transpose1d.weight.size(0)
