from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from .librosa import LibrosaMelSpectrogram

__all__ = [
    "HiFiGANMelSpectrogram",
]


class HiFiGANMelSpectrogram(LibrosaMelSpectrogram):
    """Mel-spectrogram transform for HiFi-GAN.

    Args:
        sample_rate (int): Sampling rate. Default: ``24000``.
        n_fft (int): Number of FFT bins. Default: ``1024``.
        win_length (int): Window length. By default, ``1024`` is used.
        hop_length (int): Hop length. By default, ``256`` is used.
        f_min (float): Minimum frequency of Mel-spectrogram.
        f_max (float, optional): Maximum frequency of Mel-spectrogram.
        n_mels (int): Number of mel filterbanks. Default: ``80``.
        window_fn (callable): Window function called as ``window``
            in ``librosa.feature.melspectrogram``.
        power (float, optional): Exponent for the magnitude spectrogram. Default: ``1.0``.
        pad_mode (str): Padding mode. Default: ``reflect``.
        norm (str): If ``slaney``, divide the triangular mel weights by the width of the Mel band
            (i.e. area normalization). Unlike ``torchaudio.transforms.MelSpectrogram``, defaults
            to ``slaney``.
        mel_scale (bool): Scale to use ``htk`` or ``slaney``. Unlike
            ``torchaudio.transforms.MelSpectrogram``, defaults to ``slaney``.
        take_log (bool): Whether to take log features.
        eps (bool): Epsilon to avoid log-zero.

    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 80,
        window_fn: Callable = torch.hann_window,
        power: float = 1.0,
        normalized: bool = False,
        wkwargs: Optional[Dict] = None,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = "slaney",
        mel_scale: str = "slaney",
        take_log: bool = True,
        eps: float = 1e-5,
    ) -> None:
        # to use customized padding
        center = False

        super().__init__(
            sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            pad=pad,
            n_mels=n_mels,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
            norm=norm,
            mel_scale=mel_scale,
        )

        self.take_log = take_log
        self.eps = eps

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        padding = (self.n_fft - self.hop_length) // 2
        pad_mode = self.spectrogram.pad_mode

        waveform = F.pad(waveform, [padding, padding], mode=pad_mode)
        spectrogram = super().forward(waveform)

        if self.take_log:
            spectrogram = torch.log(torch.clamp_min(spectrogram, min=self.eps))

        return spectrogram
