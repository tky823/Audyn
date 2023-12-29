from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t

from ..modules.soundstream import DecoderBlock, EncoderBlock, ResidualUnit2d
from .rvqvae import RVQVAE

__all__ = ["SoundStream", "Encoder", "Decoder", "SpectrogramDiscriminator"]


class SoundStream(RVQVAE):
    """Sound stream using residual vector quantizer.

    Args:
        encoder (nn.Module): Encoder which returns latent feature of
            shape (batch_size, embedding_dim, *).
        decoder (nn.Module): Decoder which takes latent feature of
            shape (batch_size, embedding_dim, *).
        codebook_size (int): Size of codebook.
        embedding_dim (int): Number of embedding dimension.
        num_layers (int): Number of layers of RVQ.

    """

    def forward(
        self,
        input: torch.Tensor,
        denoise: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Forward pass of RVQVAE.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, *input_shape).
            denoise (bool): If ``True``, denoising is applied.

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Reconstructed feature of same shape as input.
                - torch.Tensor: Latent feature of shape \
                    (batch_size, embedding_dim, *latent_shape). In most cases, latent_shape is \
                    smaller than input_shape.
                - torch.Tensor: Quantized latent feature of shape \
                    (batch_size, num_layers, embedding_dim, *latent_shape).
                - torch.Tensor: Indices of embeddings in codebook of shape \
                    (batch_size, num_layers, *latent_shape).

        .. note::

            Gradient from decoder does not back propagate to codebook.

        """
        if denoise:
            raise NotImplementedError("Denoising is not supported now.")

        encoded = self.encode(input)
        hierarchical_quantized, indices = self.quantize(encoded)
        quantized = hierarchical_quantized.sum(dim=1)
        quantized_straight_through = encoded + torch.detach(quantized - encoded)
        output = self.decode(quantized_straight_through, layer_wise=False)

        return output, encoded, hierarchical_quantized, indices

    @torch.no_grad()
    def inference(
        self,
        quantized: torch.Tensor,
        denoise: bool = False,
        layer_wise: bool = True,
    ) -> torch.Tensor:
        """Inference of RVQVAE.

        Args:
            quantized (torch.Tensor): Following two types are supported.
                1. Quantized latent feature of shape (batch_size, num_layers, *latent_shape)
                    or (batch_size, *latent_shape). dtype is torch.FloatTensor.
                2. Indices of quantized latent feature of shape
                    (batch_size, num_layers, *latent_shape). dtype is torch.LongTensor.
            denoise (bool): If ``True``, denoising is applied.
            layer_wise (bool): If ``True``, ``quantized`` has ``num_layers`` dimension at axis = 1.

        Returns:
            torch.Tensor: Reconstructed feature.

        """
        if denoise:
            raise NotImplementedError("Denoising is not supported now.")

        output = self.decode(quantized, layer_wise=layer_wise)

        return output


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        depth_rate: int,
        kernel_size_in: _size_1_t,
        kernel_size_out: _size_1_t,
        kernel_size: _size_1_t,
        stride: List[_size_1_t],
        dilation_rate: _size_1_t = 3,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        self.conv1d_in = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size_in,
            stride=1,
        )

        _in_channels = hidden_channels
        backbone = []

        for _stride in stride:
            _out_channels = depth_rate * _in_channels

            block = EncoderBlock(
                _in_channels,
                _out_channels,
                kernel_size=kernel_size,
                stride=_stride,
                dilation_rate=dilation_rate,
                num_layers=num_layers,
            )
            backbone.append(block)
            _in_channels = _out_channels

        self.backbone = nn.Sequential(*backbone)
        self.conv1d_out = nn.Conv1d(
            _out_channels,
            out_channels,
            kernel_size=kernel_size_out,
            stride=1,
        )

        self.kernel_size_in, self.kernel_size_out = kernel_size_in, kernel_size_out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        kernel_size_in, kernel_size_out = self.kernel_size_in, self.kernel_size_out

        x = F.pad(input, (kernel_size_in // 2, kernel_size_in // 2))
        x = self.conv1d_in(x)
        x = self.backbone(x)
        x = F.pad(x, (kernel_size_out // 2, kernel_size_out // 2))
        output = self.conv1d_out(x)

        return output


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        depth_rate: int,
        kernel_size_in: _size_1_t,
        kernel_size_out: _size_1_t,
        kernel_size: _size_1_t,
        stride: List[_size_1_t],
        dilation_rate: _size_1_t = 3,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        self.conv1d_in = nn.Conv1d(
            in_channels,
            hidden_channels ** len(stride),
            kernel_size=kernel_size_in,
            stride=1,
        )

        _in_channels = hidden_channels ** len(stride)
        backbone = []

        for _stride in stride:
            _out_channels = _in_channels // depth_rate

            block = DecoderBlock(
                _in_channels,
                _out_channels,
                kernel_size=kernel_size,
                stride=_stride,
                dilation_rate=dilation_rate,
                num_layers=num_layers,
            )
            backbone.append(block)
            _in_channels = _out_channels

        self.backbone = nn.Sequential(*backbone)
        self.conv1d_out = nn.Conv1d(
            _out_channels,
            out_channels,
            kernel_size=kernel_size_out,
            stride=1,
        )

        self.kernel_size_in, self.kernel_size_out = kernel_size_in, kernel_size_out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        kernel_size_in, kernel_size_out = self.kernel_size_in, self.kernel_size_out

        x = F.pad(input, (kernel_size_in // 2, kernel_size_in // 2))
        x = self.conv1d_in(x)
        x = self.backbone(x)
        x = F.pad(x, (kernel_size_out // 2, kernel_size_out // 2))
        output = self.conv1d_out(x)

        return output


class SpectrogramDiscriminator(nn.Module):
    """Spectrogram discriminator for SoundStream.

    Args:
        num_features (list): Number of features. ``ResidualUnit2d`` is stacked
            ``len(num_features) - 1`` times.
        kernel_size_in (_size_2_t): Kernel size of input convolution.
        kernel_size_out (_size_2_t): Kernel size of output convolution.
        kernel_size (_size_2_t): Kernel size used in ``ResidualUnit2d``.
        down_scale (list): Downsampling scale in each ``ResidualUnit2d``. The length
            should be ``len(num_features) - 1``.
        transform (bool or callable): Transform function or module to obtain spectrogram.
            If ``True`` is given, STFT is used. If callable object is given, it is expected
            as a transformation function.

    """

    def __init__(
        self,
        num_features: List[int],
        kernel_size_in: _size_2_t,
        kernel_size_out: _size_2_t,
        kernel_size: _size_2_t,
        down_scale: List[_size_2_t],
        transform: Union[bool, Callable[[torch.Tensor], torch.Tensor]] = True,
        transform_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        assert len(num_features) - 1 == len(
            down_scale
        ), "Number of items is inconsistent between num_features and down_scale."

        if type(transform) is bool:
            if transform:
                if transform_kwargs is None:
                    transform_kwargs = {}

                transform = _Spectrogram(**transform_kwargs)
            else:
                transform = None

        self.transform = transform

        self.conv2d_in = nn.Conv2d(2, num_features[0], kernel_size=kernel_size_in, stride=1)

        backbone = []

        for layer_idx in range(0, len(num_features) - 1):
            _in_channels = num_features[layer_idx]
            _out_channels = num_features[layer_idx + 1]
            _down_scale = down_scale[layer_idx]
            unit = ResidualUnit2d(
                _in_channels,
                _out_channels,
                kernel_size=kernel_size,
                down_scale=_down_scale,
            )
            backbone.append(unit)

        self.backbone = nn.Sequential(*backbone)
        self.conv2d_out = nn.Conv2d(num_features[-1], 1, kernel_size=kernel_size_out, stride=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """

        Args:
            input (torch.Tensor): Input feature. The following two features are supported.
                - waveform: (batch_size, length) or (batch_size, 1, length)
                - spectrogram: (batch_size, 2, n_bins, n_frames).

        Returns:
            torch.Tensor: Downsampled feature of shape (batch_size, 1, length').

        """
        if self.transform is None:
            x = input
        else:
            x = self.transform(input)

        x = self.conv2d_in(x)
        x = self.backbone(x)
        x = self.conv2d_out(x)

        if x.size(2) != 1:
            raise ValueError(
                "Output feature has frequency bins even after output convolution.",
            )

        batch_size, _, _, n_frames = x.size()
        output = x.view(batch_size, 1, n_frames)

        return output


class _Spectrogram(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs

        n_fft = kwargs["n_fft"]

        assert n_fft % 2 == 0, "Only even number is supported as n_fft."

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        *shape, length = waveform.size()

        assert len(shape) in [
            1,
            2,
        ], "(batch_size, length) or (batch_size, in_channels, length) is supported."

        is_monoral = True

        if len(shape) not in [1, 2]:
            msg = "(batch_size, length) or (batch_size, in_channels, length) is supported."
            is_monoral = False
        elif len(shape) == 2 and shape[1] != 1:
            msg = "Only monoral is supported now."
            is_monoral = False
        else:
            msg = None

        if not is_monoral:
            raise RuntimeError(msg)

        flatten_waveform = waveform.view(-1, length)
        spectrogram: torch.Tensor = torch.stft(flatten_waveform, **self.kwargs)

        # ignore 0th, i.e. DC (direct current) component
        _, spectrogram = torch.split(spectrogram, [1, spectrogram.size(1) - 1], dim=1)
        spectrogram = torch.view_as_real(spectrogram)
        spectrogram = spectrogram.permute(0, 3, 1, 2).contiguous()

        return spectrogram
