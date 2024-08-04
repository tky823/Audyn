from typing import Callable, Dict, Optional

import torch
import torchaudio.transforms as aT

__all__ = [
    "MusicTaggingTransformerMelSpectrogram",
]


class MusicTaggingTransformerMelSpectrogram(aT.MelSpectrogram):
    """Mel-spectrogram transform for music tagging transformer.

    For the details of arguments, see ``torchaudio.transforms.MelSpectrogram``.

    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[[int], torch.Tensor] = torch.hann_window,
        normalized: bool = False,
        wkwargs: Optional[Dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        power = 2

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