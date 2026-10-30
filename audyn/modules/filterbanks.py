import math
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t
from torchaudio.functional.functional import (
    _hz_to_mel,
    _mel_to_hz,
)

__all__ = [
    "QuadratureSincConv1d",
]


class QuadratureSincConv1d(nn.Module):
    """Learnable quadrature sinc filterbank applied with a 1D convolution.

    The filterbank has ``n_bins // 2`` learnable frequency bands. Each band
    produces an even (cosine) filter and an odd (sine) filter, for ``n_bins``
    output channels in total. Band edges are initialized uniformly on the
    selected Mel scale.

    Args:
        n_bins (int): Number of output filters. Must be an even integer of at
            least 2.
        kernel_size (int): Length of each filter. Must be odd.
        stride (int, optional): Convolution stride. Defaults to
            ``kernel_size // 2``.
        padding (int, optional): Zero-padding added to both sides of the input.
            Defaults to ``0``.
        sample_rate (float, optional): Sampling rate used to construct the
            filter frequencies. Defaults to ``16000``.
        f_min (float, optional): Minimum allowed lower edge frequency, in Hz.
            Defaults to ``50``.
        window_fn (callable, optional): Function that returns a window tensor
            of length ``kernel_size``. Defaults to a symmetric Hamming window.
        min_bandwidth (float, optional): Minimum bandwidth of each filter, in
            Hz. Defaults to ``50``.
        mel_scale (str, optional): Mel scale used to initialize the band edges.
            Defaults to ``"htk"``.
        dtype (torch.dtype, optional): Dtype used to initialize the frequency
            parameters. If omitted, PyTorch's default dtype is used.

    """

    def __init__(
        self,
        n_bins: int,
        kernel_size: _size_1_t,
        stride: Optional[int] = None,
        padding: _size_1_t = 0,
        sample_rate: float = 16000,
        f_min: float = 50,
        window_fn: Callable[[int], torch.Tensor] = partial(
            torch.hamming_window,
            periodic=False,
        ),
        min_bandwidth: float = 50,
        mel_scale: str = "htk",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        assert n_bins >= 2 and n_bins % 2 == 0
        assert kernel_size % 2 == 1
        assert sample_rate / 2 > f_min + min_bandwidth + 30
        assert f_min >= 0 and min_bandwidth > 0

        self.kernel_size = kernel_size
        self.stride = kernel_size // 2 if stride is None else stride
        self.padding = padding
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.window_fn = window_fn
        self.min_bandwidth = min_bandwidth

        window = window_fn(kernel_size)
        self.register_buffer("window", window)

        m_min = _hz_to_mel(30.0, mel_scale=mel_scale)
        m_max = _hz_to_mel(
            sample_rate / 2 - (f_min + min_bandwidth),
            mel_scale=mel_scale,
        )

        m_pts = torch.linspace(m_min, m_max, n_bins // 2 + 1, dtype=dtype)
        freqs = _mel_to_hz(m_pts, mel_scale=mel_scale)

        low_freqs, _ = torch.split(freqs, [n_bins // 2, 1], dim=0)
        _, high_freqs = torch.split(freqs, [1, n_bins // 2], dim=0)

        self.min_freqs = nn.Parameter(low_freqs)
        self.band_freqs = nn.Parameter(high_freqs - low_freqs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of QuadratureSincConv1d.

        Args:
            input (torch.Tensor): Waveform with shape ``(batch_size, 1, time)``.

        Returns:
            torch.Tensor: Filterbank features with shape ``(batch_size, n_bins, frames)``.

        """
        kernel_size = self.kernel_size
        sample_rate = self.sample_rate
        f_min = self.f_min

        device = input.device
        dtype = input.dtype

        min_freqs = f_min + torch.abs(self.min_freqs)
        max_freqs = min_freqs + self.min_bandwidth + torch.abs(self.band_freqs)
        max_freqs = torch.clamp(max_freqs, min=f_min, max=sample_rate / 2)

        center_freqs = (min_freqs + max_freqs) / 2
        band_freqs = max_freqs - min_freqs

        indices = torch.arange(kernel_size // 2 + 1, device=device, dtype=dtype)
        indices = indices / sample_rate

        x_sinc = torch.sinc(band_freqs.unsqueeze(-1) * indices)
        x = 2 * math.pi * center_freqs.unsqueeze(-1) * indices

        x_even = torch.cos(x) * x_sinc
        x_odd = torch.sin(x) * x_sinc
        _, _x_even = torch.split(x_even, [1, kernel_size // 2], dim=-1)
        _, _x_odd = torch.split(x_odd, [1, kernel_size // 2], dim=-1)
        _x_even = torch.flip(_x_even, dims=(-1,))
        _x_odd = -torch.flip(_x_odd, dims=(-1,))
        x_even = torch.cat([_x_even, x_even], dim=-1)
        x_odd = torch.cat([_x_odd, x_odd], dim=-1)

        weight = torch.cat([x_even, x_odd], dim=0)
        window = self.window.to(device=weight.device, dtype=weight.dtype)
        weight = weight * window
        output = F.conv1d(
            input,
            weight.unsqueeze(-2),
            stride=self.stride,
            padding=self.padding,
        )

        return output
