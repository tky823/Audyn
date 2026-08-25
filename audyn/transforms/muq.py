import os
from typing import Optional

import torch

from ..utils._github import download_file_from_github_release
from .musicfm import MusicFMMelSpectrogram


class MuQMelSpectrogram(MusicFMMelSpectrogram):
    @classmethod
    def build_from_pretrained(
        cls,
        dataset: Optional[str] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> "MuQMelSpectrogram":
        """Build MuQMelSpectrogram from pretraind one.

        Due to lack of backward compatibility of ``torchaudio.transforms.MelSpectrogram``,
        you need to use ``build_from_pretrained`` for official implementation.

        Examples:

            >>> import torch
            >>> from audyn.transforms import MuQMelSpectrogram
            >>> torch.manual_seed(0)
            >>> transform = MuQMelSpectrogram.build_from_pretrained(dataset="fma")
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
                - msd

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

            url = "https://github.com/tky823/Audyn/releases/download/v0.2.0/muq_melspectrogram.pth"  # noqa: E501
            path = os.path.join(model_cache_dir, "MuQ", "c6a0b236", "muq_melspectrogram.pth")
            download_file_from_github_release(url, path=path)

            state_dict = torch.load(
                path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )

            if dataset.lower() == "msd":
                mean = 6.768444971712967
                std = 18.417922652295623
            else:
                raise ValueError(f"Unsupported dataset {dataset} is given.")

            transform = cls(mean=mean, std=std)
            transform.load_state_dict(state_dict)

        return transform
