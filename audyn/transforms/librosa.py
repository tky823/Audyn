from typing import Callable, Dict, Optional

import torch
import torchaudio.transforms as aT

from ..functional.melspectrogram import melscale_fbanks

__all__ = [
    "LibrosaMelSpectrogram",
]


class LibrosaMelSpectrogram(aT.MelSpectrogram):
    """Mel-spectrogram transform compatible with ``librosa.feature.melspectrogram``.

    Args:
        sample_rate (int): Sampling rate. Default: ``22050``.
        n_fft (int): Number of FFT bins. Default: ``2048``.
        win_length (int): Window length. By default, ``n_fft`` is used.
        hop_length (int): Hop length. By default, ``512`` is used.
        f_min (float): Minimum frequency of Mel-spectrogram.
        f_max (float, optional): Maximum frequency of Mel-spectrogram.
        n_mels (int): Number of mel filterbanks. Default: ``128``.
        window_fn (callable): Window function called as ``window``
            in ``librosa.feature.melspectrogram``.
        power (float, optional): Exponent for the magnitude spectrogram.
        center (bool): whether to pad waveform on both sides so that the t-th
            frame is centered at corresponding frame.
        pad_mode (str): Padding mode when ``center=True``. Unlike
            ``torchaudio.transforms.MelSpectrogram``, defaults to ``constant``.
        norm (str): If ``slaney``, divide the triangular mel weights by the width of the Mel band
            (i.e. area normalization). Unlike ``torchaudio.transforms.MelSpectrogram``, defaults
            to ``slaney``.
        mel_scale (bool): Scale to use ``htk`` or ``slaney``. Unlike
            ``torchaudio.transforms.MelSpectrogram``, defaults to ``slaney``.

    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        win_length: Optional[int] = None,
        hop_length: int = 512,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[Dict] = None,
        center: bool = True,
        pad_mode: str = "constant",
        onesided: Optional[bool] = None,
        norm: Optional[str] = "slaney",
        mel_scale: str = "slaney",
        fb_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if win_length is None:
            win_length = n_fft

        if f_max is None:
            f_max = sample_rate / 2

        is_float64_fb = fb_dtype is torch.float64

        if is_float64_fb:
            if wkwargs is None:
                wkwargs = {}

            wkwargs["dtype"] = torch.float64

        super().__init__(
            sample_rate=sample_rate,
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

        self.fb_dtype = fb_dtype

        if is_float64_fb:
            fb = melscale_fbanks(
                n_fft // 2 + 1,
                f_min,
                f_max,
                n_mels=n_mels,
                sample_rate=sample_rate,
                norm=norm,
                mel_scale=mel_scale,
                dtype=torch.float64,
            )

            self.mel_scale.fb = fb

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform_dtype = waveform.dtype
        waveform = waveform.to(self.mel_scale.fb.dtype)
        output = super().forward(waveform)
        output = output.to(waveform_dtype)

        return output
