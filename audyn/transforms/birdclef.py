from typing import Callable, Dict, Optional, Tuple

import torch
import torchaudio
import torchaudio.functional as aF
from packaging import version

from .librosa import LibrosaMelSpectrogram

__all__ = [
    "BirdCLEF2024BaselineMelSpectrogram",
]

IS_TORCHAUDIO_LT_2_1 = version.parse(torchaudio.__version__) < version.parse("2.1")


class BirdCLEF2024BaselineMelSpectrogram(LibrosaMelSpectrogram):
    """Mel-spectrogram transform for BirdCLEF2024.

    Args:
        sample_rate (int): Sampling rate. Default: ``32000``.
        n_fft (int): Number of FFT bins. Default: ``2048``.
        win_length (int): Window length. By default, ``n_fft`` is used.
        hop_length (int): Hop length. By default, ``1253`` is used.
        f_min (float): Minimum frequency of Mel-spectrogram. Default: ``20``.
        f_max (float, optional): Maximum frequency of Mel-spectrogram.
        n_mels (int): Number of mel filterbanks. Default: ``128``.
        window_fn (callable): Window function called as ``window``
            in ``librosa.feature.melspectrogram``.
        power (float, optional): Exponent for the magnitude spectrogram. Default: ``1``.
        center (bool): whether to pad waveform on both sides so that the t-th
            frame is centered at corresponding frame.
        pad_mode (str): Padding mode when ``center=True``. Unlike
            ``torchaudio.transforms.MelSpectrogram``, defaults to ``constant``.
        norm (str): If ``slaney``, divide the triangular mel weights by the width of the Mel band
            (i.e. area normalization). Unlike ``torchaudio.transforms.MelSpectrogram``, defaults
            to ``slaney``.
        mel_scale (bool): Scale to use ``htk`` or ``slaney``. Unlike
            ``torchaudio.transforms.MelSpectrogram``, defaults to ``slaney``.
        take_log (bool): Whether to return log-melspectrogram or melspectrogram.
        freq_mask_param (tuple): Parameter of frequency masking. Default: ``(0.06, 0.1)``.
        time_mask_param (tuple): Parameter of time masking. Default: ``(0.06, 0.12)``.
        eps (float, optional): Epsilon to avoid log-zero. If ``take_log=True``,
            ``eps`` defaults to ``1e-10``.

    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 2048,
        win_length: Optional[int] = None,
        hop_length: int = 1253,
        f_min: float = 20.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable = torch.hann_window,
        power: float = 1.0,
        normalized: bool = False,
        wkwargs: Optional[Dict] = None,
        center: bool = True,
        pad_mode: str = "constant",
        onesided: Optional[bool] = None,
        norm: Optional[str] = "slaney",
        mel_scale: str = "slaney",
        take_log: bool = True,
        freq_mask_param: Tuple[float, float] = (0.06, 0.1),
        time_mask_param: Tuple[float, float] = (0.06, 0.12),
        eps: Optional[float] = None,
    ) -> None:
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

        self.take_log = take_log
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

        assert 0 <= freq_mask_param[0] <= 1
        assert 0 <= freq_mask_param[1] <= 1
        assert 0 <= time_mask_param[0] <= 1
        assert 0 <= time_mask_param[1] <= 1

        if take_log and eps is None:
            eps = 1e-10

        self.eps = eps

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        specgram = super().forward(waveform)

        if self.take_log:
            if self.eps is not None:
                specgram = torch.clamp(specgram, min=self.eps)

            specgram = torch.log(specgram)

        # normalization by std-mean
        std, mean = torch.std_mean(specgram, dim=(-2, -1), unbiased=False, keepdim=True)
        std = std.masked_fill(std == 0, 1)
        specgram = (specgram - mean) / std

        # normalization by min-max
        *batch_shape, n_bins, n_frames = specgram.size()
        specgram = specgram.contiguous()
        specgram = specgram.view(*batch_shape, n_bins * n_frames)
        vmin, _ = torch.min(specgram, dim=-1, keepdim=True)
        vmax, _ = torch.max(specgram, dim=-1, keepdim=True)
        vrange = vmax - vmin
        vrange = vrange.masked_fill(vrange == 0, 1)
        specgram = (specgram - vmin) / vrange
        specgram = specgram.view(*batch_shape, n_bins, n_frames)

        if self.training:
            freq_mask_param_min, freq_mask_param_max = self.freq_mask_param
            time_mask_param_min, time_mask_param_max = self.time_mask_param

            freq_mask_param_min = int(n_bins * freq_mask_param_min)
            freq_mask_param_max = int(n_bins * freq_mask_param_max)
            time_mask_param_min = int(n_frames * time_mask_param_min)
            time_mask_param_max = int(n_frames * time_mask_param_max)

            freq_mask_param = torch.randint(freq_mask_param_min, freq_mask_param_max + 1, ())
            time_mask_param = torch.randint(time_mask_param_min, time_mask_param_max + 1, ())
            freq_mask_param = freq_mask_param.item()
            time_mask_param = time_mask_param.item()

            if IS_TORCHAUDIO_LT_2_1:
                # 4D is required.
                specgram = specgram.view(-1, 1, n_bins, n_frames)
                freq_axis = 2
                time_axis = 3
            else:
                # 3D is required.
                specgram = specgram.view(-1, n_bins, n_frames)
                freq_axis = 1
                time_axis = 2

            # frequency mask
            specgram = aF.mask_along_axis_iid(
                specgram, mask_param=freq_mask_param, mask_value=0, axis=freq_axis
            )

            # time mask
            specgram = aF.mask_along_axis_iid(
                specgram, mask_param=time_mask_param, mask_value=0, axis=time_axis
            )

            specgram = specgram.view(*batch_shape, n_bins, n_frames)

        return specgram
