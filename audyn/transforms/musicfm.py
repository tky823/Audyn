import os
from typing import Callable, Dict, Optional

import torch
import torchaudio.transforms as aT
from packaging import version

from ..amp import autocast, get_autocast_device_type
from ..utils._github import download_file_from_github_release

__all__ = [
    "MusicFMMelSpectrogram",
]

IS_TORCH_LT_2_3 = version.parse(torch.__version__) < version.parse("2.3")


class MusicFMMelSpectrogram(aT.MelSpectrogram):
    """Mel-spectrogram transform for MusicFM.

    .. note::

        It is recommended to use ``MusicFMMelSpectrogram.build_from_pretrained``
        if you reproduce official module.

    Args:
        sample_rate (int): Sampling rate. Default: ``24000``.
        n_fft (int): Number of FFT bins. Default: ``2048``.
        win_length (int): Window length. By default, ``n_fft`` is used.
        hop_length (int): Hop length. By default, ``240`` is used.
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
            (i.e. area normalization).
        mel_scale (bool): Scale to use ``htk`` or ``slaney``. Default: ``htk``.
        dtype (torch.dtype, optional): Data type during forward and backward pass.

    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 2048,
        win_length: Optional[int] = None,
        hop_length: int = 240,
        f_min: float = 0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[[int], torch.Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[Dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        mean: float = 0,
        std: float = 1.0,
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
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

        self.amplitude_to_db = aT.AmplitudeToDB()

        self.mean = mean
        self.std = std
        self.dtype = dtype

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass of MelSpectrogram.

        Args:
            waveform (torch.Tensor): Waveform of shape (*, timesteps).

        Returns:
            torch.Tensor: Mel-spectrogram of shape (*, n_mels, n_frames).

        """
        dtype = self.dtype
        device_type = get_autocast_device_type()

        if dtype is torch.float32 or dtype is torch.float64:
            enabled = False
        elif dtype is torch.float16 or dtype is torch.bfloat16:
            enabled = True
        else:
            raise ValueError(
                f"Unsupported dtype {dtype} is given. Use float32, float16, or bfloat16."
            )

        if IS_TORCH_LT_2_3 and not enabled:
            spectrogram = self._forward(waveform)
        else:
            with autocast(device_type=device_type, dtype=dtype, enabled=enabled):
                spectrogram = self._forward(waveform)

        spectrogram = (spectrogram - self.mean) / self.std

        return spectrogram

    def _forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass of MelSpectrogram without normalization.

        Args:
            waveform (torch.Tensor): Waveform of shape (*, timesteps).

        Returns:
            torch.Tensor: Mel-spectrogram of shape (*, n_mels, n_frames).

        """
        spectrogram = super().forward(waveform)
        spectrogram, _ = torch.split(spectrogram, [spectrogram.size(-1) - 1, 1], dim=-1)
        spectrogram = self.amplitude_to_db(spectrogram)

        return spectrogram

    @classmethod
    def build_from_pretrained(
        cls,
        dataset: Optional[str] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> "MusicFMMelSpectrogram":
        """Build MusicFMMelSpectrogram from pretraind one.

        Due to lack of backward compatibility of ``torchaudio.transforms.MelSpectrogram``,
        you need to use ``build_from_pretrained`` for official implementation.

        Examples:

            >>> import torch
            >>> from audyn.transforms import MusicFMMelSpectrogram
            >>> torch.manual_seed(0)
            >>> transform = MusicFMMelSpectrogram.build_from_pretrained(dataset="fma")
            >>> sample_rate = transform.sample_rate
            >>> print(sample_rate)
            24000
            >>> waveform = torch.randn((30 * sample_rate))
            >>> spectrogram = transform(waveform)
            >>> print(spectrogram.size())
            torch.Size([128, 3000])

        .. note::

            Supported pretrained model names are
                - fma
                - musicfm_msd

        """
        from ..utils import model_cache_dir

        if dataset is None:
            if mean is None:
                mean = 0

            if std is None:
                std = 1

            transform = cls(mean=mean, std=std)
        else:
            assert mean is None and std is None, "mean and std should be None."

            url = "https://github.com/tky823/Audyn/releases/download/v0.2.0/musicfm_melspectrogram.pth"  # noqa: E501
            path = os.path.join(
                model_cache_dir, "MusicFM", "72f9d23e", "musicfm_melspectrogram.pth"
            )
            download_file_from_github_release(url, path=path)

            state_dict = torch.load(
                path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )

            if dataset.lower() == "fma":
                mean = 3.0710578151459664
                std = 20.999089814337626
            elif dataset.lower() == "msd":
                mean = 6.768444971712967
                std = 18.417922652295623
            else:
                raise ValueError(f"Unsupported dataset {dataset} is given.")

            transform = cls(mean=mean, std=std)
            transform.load_state_dict(state_dict)

        return transform
