from typing import Callable, Dict, Optional

import torch
import torchaudio.transforms as aT

__all__ = [
    "LAIONAudioEncoder2023MelSpectrogram",
]


class LAIONAudioEncoder2023MelSpectrogram(aT.MelSpectrogram):
    """Mel-spectrogram transform for LAIONAudioEncoder2023.

    For the details of arguments, see ``torchaudio.transforms.MelSpectrogram``.

    .. note::

        Value of ``mel_scale`` depends on ``truncation`` in original implementation.
        See https://github.com/huggingface/transformers/blob/6bc0fbcfa7acb6ac4937e7456a76c2f7975fefec/src/transformers/models/clap/feature_extraction_clap.py#L153-L163.

    Examples:

        >>> import torch
        >>> from audyn.transforms import LAIONAudioEncoder2023MelSpectrogram
        >>> torch.manual_seed(0)
        >>> waveform = torch.randn((48000,))
        >>> # truncation is "fusion"
        >>> melspectrogram_transform = LAIONAudioEncoder2023MelSpectrogram(norm=None, mel_scale="htk")
        >>> melspectrogram = melspectrogram_transform(waveform)
        >>> melspectrogram.size()
        torch.Size([64, 101])
        >>> # truncation is NOT "fusion" (default by transformers)
        >>> # https://github.com/huggingface/transformers/blob/6bc0fbcfa7acb6ac4937e7456a76c2f7975fefec/src/transformers/models/clap/feature_extraction_clap.py#L96
        >>> melspectrogram_transform = LAIONAudioEncoder2023MelSpectrogram(norm="slaney", mel_scale="slaney")
        >>> melspectrogram = melspectrogram_transform(waveform)
        >>> melspectrogram.size()
        torch.Size([64, 101])

    """  # noqa: E501

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 480,
        f_min: float = 50,
        f_max: float = 14000,
        pad: int = 0,
        n_mels: int = 64,
        window_fn: Callable[[int], torch.Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[Dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
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

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram = super().forward(waveform)
        output = self.amplitude_to_db(spectrogram)

        return output
