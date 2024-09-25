import os
from typing import Callable, Dict, Optional

import torch
import torchaudio.transforms as aT

from ..utils.github import download_file_from_github_release

__all__ = [
    "MusicTaggingTransformerMelSpectrogram",
]


class MusicTaggingTransformerMelSpectrogram(aT.MelSpectrogram):
    """Mel-spectrogram transform for music tagging transformer.

    For the details of arguments, see ``torchaudio.transforms.MelSpectrogram``.

    .. note::

        It is recommended to use ``MusicTaggingTransformerMelSpectrogram.build_from_pretrained``
        if you reproduce official module.

    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
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

    @classmethod
    def build_from_pretrained(cls) -> "MusicTaggingTransformerMelSpectrogram":
        """Build MusicTaggingTransformerMelSpectrogram from pretraind one.

        Due to lack of backward compatibility of ``torchaudio.transforms.MelSpectrogram``,
        you need to use ``build_from_pretrained`` for official implementation.

        Examples:

            >>> import torch
            >>> from audyn.transforms import MusicTaggingTransformerMelSpectrogram
            >>> torch.manual_seed(0)
            >>> transform = MusicTaggingTransformerMelSpectrogram.build_from_pretrained()
            >>> sample_rate = transform.sample_rate
            >>> print(sample_rate)
            22050
            >>> waveform = torch.randn((30 * sample_rate))
            >>> spectrogram = transform(waveform)
            >>> print(spectrogram.size())
            torch.Size([128, 1292])

        """
        from ..utils import model_cache_dir

        sample_rate = 22050

        url = "https://github.com/tky823/Audyn/releases/download/v0.0.2/music-tagging-transformer_melspectrogram.pth"  # noqa: E501
        path = os.path.join(
            model_cache_dir,
            "MusicTaggingTransformer",
            "music-tagging-transformer_melspectrogram.pth",
        )
        download_file_from_github_release(url, path=path)

        state_dict = torch.load(path, map_location=lambda storage, loc: storage)

        transform = cls(sample_rate)
        transform.load_state_dict(state_dict)

        return transform
