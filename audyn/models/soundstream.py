from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t
from torch.nn.modules.utils import _single

from ..modules.soundstream import DecoderBlock, EncoderBlock, FiLM1d, ResidualUnit2d
from .hifigan import MultiScaleDiscriminator as _MultiScaleDiscriminator
from .hifigan import ScaleDiscriminator as _ScaleDiscriminator
from .rvqvae import RVQVAE

__all__ = [
    "SoundStream",
    "SoundStreamReconstructor",
    "Discriminator",
    "Encoder",
    "Decoder",
    "MultiScaleDiscriminator",
    "ScaleDiscriminator",
    "SpectrogramDiscriminator",
]


class SoundStream(RVQVAE):
    """Sound stream using residual vector quantizer.

    Args:
        encoder (nn.Module): Encoder which returns latent feature of
            shape (batch_size, embedding_dim, *).
        decoder (nn.Module): Decoder which takes latent feature of
            shape (batch_size, embedding_dim, *).
        codebook_size (int): Size of codebook.
        embedding_dim (int): Number of embedding dimension.
        num_stages (int): Number of residual stages of RVQ.
        dropout (bool): Dropout of RVQ. Default: ``True``.
        init_by_kmeans (int): Number of iterations in k-means clustering initialization.
            If non-positive value is given, k-means clustering initialization is not used.
        seed (int): Random seed for k-means clustering initialization.

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
                    (batch_size, num_stages, embedding_dim, *latent_shape).
                - torch.Tensor: Indices of embeddings in codebook of shape \
                    (batch_size, num_stages, *latent_shape).

        .. note::

            Gradient from decoder does not back propagate to codebook.

        """
        if denoise:
            raise NotImplementedError("Denoising is not supported now.")

        padding_left, padding_right = self.compute_padding(input)
        x = F.pad(input, (padding_left, padding_right))
        encoded = self.encode(x)
        hierarchical_quantized, indices = self.quantize(encoded)
        quantized = hierarchical_quantized.sum(dim=1)
        quantized_straight_through = encoded + torch.detach(quantized - encoded)
        x = self.decode(quantized_straight_through, stage_wise=False)
        output = F.pad(x, (-padding_left, -padding_right))

        return output, encoded, hierarchical_quantized, indices

    @torch.no_grad()
    def inference(
        self,
        quantized: torch.Tensor,
        denoise: bool = False,
        stage_wise: bool = True,
    ) -> torch.Tensor:
        """Inference of RVQVAE.

        Args:
            quantized (torch.Tensor): Following two types are supported.
                1. Quantized latent feature of shape (batch_size, num_stages, *latent_shape)
                    or (batch_size, *latent_shape). dtype is torch.FloatTensor.
                2. Indices of quantized latent feature of shape
                    (batch_size, num_stages, *latent_shape). dtype is torch.LongTensor.
            denoise (bool): If ``True``, denoising is applied.
            stage_wise (bool): If ``True``, ``quantized`` has ``num_stages`` dimension at axis = 1.

        Returns:
            torch.Tensor: Reconstructed feature.

        """
        if denoise:
            raise NotImplementedError("Denoising is not supported now.")

        output = self.decode(quantized, stage_wise=stage_wise)

        return output

    def compute_padding(self, input: torch.Tensor) -> Tuple[int, int]:
        """Compute padding size based on down_scale.

        Args:
            input (torch.Tensor): Waveform-like feature of shape (batch_size, in_channels, length).

        Returns:
            tuple: Padding size of left- and right-hand side.

        """
        down_scale = self.down_scale

        if down_scale is None:
            padding = 0
        else:
            length = input.size(-1)
            padding = (down_scale - length % down_scale) % down_scale

        padding_left = padding // 2
        padding_right = padding - padding_left

        return padding_left, padding_right

    @property
    def down_scale(self) -> Optional[int]:
        """Try to find down_scale or downscale parameter from self.encoder."""
        encoder = self.encoder
        scale = 1

        if isinstance(encoder, Encoder):
            encoder: Encoder

            for s in encoder.stride:
                scale *= s
        elif hasattr(encoder, "down_scale") and not callable(encoder.down_scale):
            scale = encoder.down_scale
        elif hasattr(encoder, "downscale") and not callable(encoder.downscale):
            scale = encoder.downscale
        else:
            scale = None

        return scale

    @property
    def up_scale(self) -> Optional[int]:
        """Try to find up_scale or upscale parameter from self.decoder."""
        decoder = self.decoder
        scale = 1

        if isinstance(decoder, Decoder):
            decoder: Decoder

            for s in decoder.stride:
                scale *= s
        elif hasattr(decoder, "up_scale") and not callable(decoder.up_scale):
            scale = decoder.up_scale
        elif hasattr(decoder, "upscale") and not callable(decoder.upscale):
            scale = decoder.upscale
        else:
            scale = None

        return scale


class SoundStreamReconstructor(SoundStream):
    """Wrapper class of SoundStream for waveform reconstruction.

    Unlike SoundStream class, inference method is used for reconstruction.
    """

    @torch.no_grad()
    def inference(
        self,
        input: torch.Tensor,
        denoise: bool = False,
        num_stages: Optional[int] = None,
    ) -> torch.Tensor:
        """Inference of SoundStreamReconstructor.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, *input_shape).
            denoise (bool): If ``True``, denoising is applied.

        Returns:
            torch.Tensor: Reconstructed feature of same shape as input.

        """
        if denoise:
            raise NotImplementedError("Denoising is not supported now.")

        encoded = self.encode(input)
        hierarchical_quantized, _ = self.quantize(encoded)

        num_total_stages = hierarchical_quantized.size(1)

        if num_stages is None:
            num_stages = num_total_stages

        hierarchical_quantized, _ = torch.split(
            hierarchical_quantized, [num_stages, num_total_stages - num_stages], dim=1
        )
        quantized = hierarchical_quantized.sum(dim=1)
        output = self.decode(quantized, stage_wise=False)

        return output


class Discriminator(nn.Module):
    """Discriminator for SoundStream."""

    def __init__(
        self,
        waveform_discriminator: "MultiScaleDiscriminator",
        spectrogram_discriminator: "SpectrogramDiscriminator",
    ) -> None:
        super().__init__()

        self.waveform_discriminator = waveform_discriminator
        self.spectrogram_discriminator = spectrogram_discriminator

    def forward(self, input: torch.Tensor) -> Tuple[Any, Any]:
        """Forward pass of Discriminator.

        Args:
            input (torch.Tensor): Waveform of shape (batch_size, length) or
                (batch_size, in_channels, length).

        Returns:
            tuple: Tuple of tensors containing

                - Output of ``waveform_discriminator``.
                - Output of ``spectrogram_discriminator``.

        """
        waveform_output = self.waveform_discriminator(input)
        spectrogram_output = self.spectrogram_discriminator(input)

        return waveform_output, spectrogram_output


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
        is_causal: bool = True,
    ) -> None:
        super().__init__()

        self.conv1d_in = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size_in,
            stride=1,
        )
        self.nonlinear1d = nn.ELU()

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
                is_causal=is_causal,
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

        # num_modes
        # 0: reconstruct, 1: denoise
        num_modes = 2
        self.film1d = FiLM1d(num_modes, out_channels)

        self.kernel_size_in = _single(kernel_size_in)
        self.kernel_size_out = _single(kernel_size_out)
        self.stride = stride
        self.is_causal = is_causal

    def forward(self, input: torch.Tensor, denoise: bool = False) -> torch.Tensor:
        """Forward pass of Encoder.

        Args:
            input (torch.Tensor): Waveform of shape (batch_size, in_channels, length).
            denoise (bool): Denoising flag.

        Returns:
            torch.Tensor: Compressed feature of shape (batch_size, out_channels, length').

        """
        (kernel_size_in,), (kernel_size_out,) = self.kernel_size_in, self.kernel_size_out
        batch_size = input.size(0)

        if denoise:
            fill_value = 1
        else:
            fill_value = 0

        mode = torch.full(
            (batch_size,), fill_value=fill_value, dtype=torch.long, device=input.device
        )

        x = self._pad1d(input, kernel_size=kernel_size_in)
        x = self.conv1d_in(x)
        x = self.nonlinear1d(x)
        x = self.backbone(x)
        x = self._pad1d(x, kernel_size=kernel_size_out)
        x = self.conv1d_out(x)
        output = self.film1d(x, mode=mode)

        return output

    def _pad1d(self, input: torch.Tensor, kernel_size: _size_1_t) -> torch.Tensor:
        # assume stride = 1
        (kernel_size,) = _single(kernel_size)
        padding = kernel_size - 1

        if self.is_causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        output = F.pad(input, (padding_left, padding_right))

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
        is_causal: bool = True,
    ) -> None:
        super().__init__()

        # num_modes
        # 0: reconstruct, 1: denoise
        num_modes = 2
        self.film1d = FiLM1d(num_modes, in_channels)
        self.conv1d_in = nn.Conv1d(
            in_channels,
            hidden_channels * (depth_rate ** len(stride)),
            kernel_size=kernel_size_in,
            stride=1,
        )
        self.nonlinear1d = nn.ELU()

        _in_channels = hidden_channels * (depth_rate ** len(stride))
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
                is_causal=is_causal,
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
        self.tanh = nn.Tanh()

        self.kernel_size_in = _single(kernel_size_in)
        self.kernel_size_out = _single(kernel_size_out)
        self.is_causal = is_causal

    def forward(self, input: torch.Tensor, denoise: bool = False) -> torch.Tensor:
        """Forward pass of Decoder.

        Args:
            input (torch.Tensor): Compressed feature of shape (batch_size, in_channels, length).
            denoise (bool): Denoising flag.

        Returns:
            torch.Tensor: Reconstructed waveform of shape (batch_size, out_channels, length').

        """
        (kernel_size_in,), (kernel_size_out,) = self.kernel_size_in, self.kernel_size_out
        batch_size = input.size(0)

        if denoise:
            fill_value = 1
        else:
            fill_value = 0

        mode = torch.full(
            (batch_size,), fill_value=fill_value, dtype=torch.long, device=input.device
        )

        x = self.film1d(input, mode=mode)
        x = self._pad1d(x, kernel_size=kernel_size_in)
        x = self.conv1d_in(x)
        x = self.nonlinear1d(x)
        x = self.backbone(x)
        x = self._pad1d(x, kernel_size=kernel_size_out)
        x = self.conv1d_out(x)
        output = self.tanh(x)

        return output

    def _pad1d(self, input: torch.Tensor, kernel_size: _size_1_t) -> torch.Tensor:
        # assume stride = 1
        (kernel_size,) = _single(kernel_size)
        padding = kernel_size - 1

        if self.is_causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        output = F.pad(input, (padding_left, padding_right))

        return output


class MultiScaleDiscriminator(_MultiScaleDiscriminator):
    """Multi-scale discriminator.

    Args:
        num_features (list): Number of features in convolution. This value is given to
            each sub-discriminator.
        kernel_size (list): Kernel sizes in convolution. This value is given to
            each sub-discriminator.
        stride (list or _size_1_t): Strides in convolution. This value is given to
            each sub-discriminator.
        dilation (list or _size_1_t): Dilation factor in convolution. This value is given to
            each sub-discriminator.
        groups (list or int): List of groupds in convolution. This value is given to
            each sub-discriminator.
        negative_slope (float): Negative slope in leaky relu.
        pool_kernel_size (_size_1_t): Kernel size in pooling layer.
        pool_stride (_size_1_t): Stride in pooling layer.
        weight_regularization (list, optional): List of weight regularization methods,
            whose length corresponds to number of sub-discriminators.
            Only ``weight_norm`` and ``spectral_norm`` are supported.

    """

    @classmethod
    def build_from_default_config(cls) -> "MultiScaleDiscriminator":
        """Build multi-scale discriminator from default config.

        Returns:
            MultiScaleDiscriminator: Multi-scale discriminator by default parameters.

        """
        num_features = [1, 4, 16, 64, 256, 1024, 1024]
        kernel_size = [15, 41, 41, 41, 41, 5, 3]
        stride = [1, 4, 4, 4, 4, 1, 1]
        dilation = [1, 1, 1, 1, 1, 1, 1]
        groups = [1, 4, 16, 64, 256, 1, 1]
        negative_slope = 0.1
        pool_kernel_size = 4
        pool_stride = 2
        weight_regularization = [None, None, None]

        discriminator = cls(
            num_features,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            negative_slope=negative_slope,
            pool_kernel_size=pool_kernel_size,
            pool_stride=pool_stride,
            weight_regularization=weight_regularization,
        )

        return discriminator


class ScaleDiscriminator(_ScaleDiscriminator):
    """Scale discriminator for SoundStream.

    Args:
        num_features (list): Number of features in convolution.
        kernel_size (list): Kernel sizes in convolution.
        stride (list or _size_1_t): Strides in convolution.
        dilation (list or _size_1_t): Dilation factor in convolution.
        groups (list or int): Number of groupds in convolution.
        negative_slope (float): Negative slope in leaky relu.
        weight_regularization (list, optional): Weight regularization method.
            Only ``weight_norm`` and ``spectral_norm`` are supported.

    """

    @classmethod
    def build_from_default_config(
        cls, weight_regularization: Optional[str] = None
    ) -> "ScaleDiscriminator":
        """Build scale discriminator from default config.

        Returns:
            ScaleDiscriminator: Scale discriminator by default parameters.

        """
        num_features = [1, 4, 16, 64, 256, 1024, 1024]
        kernel_size = [15, 41, 41, 41, 41, 5, 3]
        stride = [1, 4, 4, 4, 4, 1, 1]
        dilation = [1, 1, 1, 1, 1, 1, 1]
        groups = [1, 4, 16, 64, 256, 1, 1]
        negative_slope = 0.1

        discriminator = cls(
            num_features,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            negative_slope=negative_slope,
            weight_regularization=weight_regularization,
        )

        return discriminator


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

        self.backbone = nn.ModuleList(backbone)
        self.conv2d_out = nn.Conv2d(num_features[-1], 1, kernel_size=kernel_size_out, stride=1)

    @classmethod
    def build_from_default_config(cls) -> "SpectrogramDiscriminator":
        num_features = [32, 32, 64, 128, 128, 256, 256]
        kernel_size_in, kernel_size_out, kernel_size = (7, 7), (8, 1), 3
        down_scale = [(2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2)]
        transform = _Spectrogram(n_fft=1024, hop_length=256, return_complex=True)

        return cls(
            num_features,
            kernel_size_in=kernel_size_in,
            kernel_size_out=kernel_size_out,
            kernel_size=kernel_size,
            down_scale=down_scale,
            transform=transform,
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass of SpectrogramDiscriminator.

        Args:
            input (torch.Tensor): Input feature. The following two features are supported.
                - waveform: (batch_size, length) or (batch_size, 1, length)
                - spectrogram: (batch_size, 2, n_bins, n_frames).

        Returns:
            tuple: Tuple of tensors containing:

                - torch.Tensor: Downsampled feature of shape (batch_size, 1, length').
                - list: Feature maps of length ``len(num_features) - 1``.

        """
        if self.transform is None:
            x = input
        else:
            x = self.transform(input)

        x = self.conv2d_in(x)
        feature_map = []

        for unit in self.backbone:
            x = unit(x)
            feature_map.append(x)

        x = self.conv2d_out(x)

        if x.size(2) != 1:
            raise ValueError(
                "Output feature has frequency bins even after output convolution.",
            )

        batch_size, _, _, n_frames = x.size()
        output = x.view(batch_size, 1, n_frames)

        return output, feature_map


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
