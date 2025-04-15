from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as aT

__all__ = [
    "LAIONCLAPAudioEncoder2023MelSpectrogram",
    "LAIONAudioEncoder2023MelSpectrogram",
    "LAIONCLAPAudioEncoder2023MelSpectrogramFusion",
    "LAIONAudioEncoder2023MelSpectrogramFusion",
]


class LAIONCLAPAudioEncoder2023MelSpectrogram(aT.MelSpectrogram):
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


class LAIONCLAPAudioEncoder2023MelSpectrogramFusion(nn.Module):
    """Mel-spectrogram fusion for LAIONAudioEncoder2023.

    Args:
        chunk_size (int): Chunk size. Default: ``480000``.
        num_chunks (int): Number of chunks. Default: ``3``.
        dim (int): Dimension to stack. Default: ``-3``.
        prepend_resampled_chunk (bool): If ``True``, resampled chunk of Mel-spectrogram
            is prepended.
        sample_wise (bool): If ``True``, masking is applied per sample.

    """

    def __init__(
        self,
        chunk_size: int = 1001,
        num_chunks: int = 3,
        dim: int = -3,
        prepend_resampled_chunk: bool = True,
        pad_mode: str = "replicate+constant",
        sample_wise: bool = True,
    ) -> None:
        super().__init__()

        assert pad_mode in ["replicate", "replicate+constant"]

        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.dim = dim
        self.prepend_resampled_chunk = prepend_resampled_chunk
        self.pad_mode = pad_mode
        self.sample_wise = sample_wise

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        chunk_size = self.chunk_size
        num_chunks = self.num_chunks

        assert spectrogram.dim() >= 2

        *batch_shape, n_bins, n_frames = spectrogram.size()
        spectrogram = spectrogram.view(-1, n_bins, n_frames)

        if n_frames >= chunk_size:
            if self.sample_wise:
                stacked_chunks = [[] for _ in range(num_chunks)]

                for _spectrogram in spectrogram:
                    _spectrogram = self._sample_chunks(_spectrogram)

                    for chunk_idx in range(num_chunks):
                        stacked_chunks[chunk_idx].append(_spectrogram[chunk_idx])

                for chunk_idx in range(num_chunks):
                    stacked_chunks[chunk_idx] = torch.stack(stacked_chunks[chunk_idx], dim=0)
                    stacked_chunks[chunk_idx] = stacked_chunks[chunk_idx].view(
                        *batch_shape, n_bins, chunk_size
                    )
            else:
                stacked_chunks = self._sample_chunks(spectrogram)

                for chunk_idx in range(num_chunks):
                    stacked_chunks[chunk_idx] = stacked_chunks[chunk_idx].view(
                        *batch_shape, n_bins, chunk_size
                    )
        else:
            stacked_chunks = self._pad_chunks(spectrogram)

            for chunk_idx in range(num_chunks):
                stacked_chunks[chunk_idx] = stacked_chunks[chunk_idx].view(
                    *batch_shape, n_bins, chunk_size
                )

        output = torch.stack(stacked_chunks, dim=self.dim)

        if self.prepend_resampled_chunk:
            resampled_chunk = F.interpolate(
                spectrogram.view(-1, 1, n_bins, n_frames),
                size=(n_bins, chunk_size),
                mode="bilinear",
                align_corners=False,
            )
            resampled_chunk = resampled_chunk.view(*batch_shape, n_bins, chunk_size)
            resampled_chunk = resampled_chunk.unsqueeze(dim=self.dim)
            output = torch.cat([resampled_chunk, output], dim=self.dim)

        return output

    def _sample_chunks(self, spectrogram: torch.Tensor) -> List[torch.Tensor]:
        chunk_size = self.chunk_size
        num_chunks = self.num_chunks
        n_frames = spectrogram.size(-1)
        valid_max_idx = n_frames - chunk_size

        if valid_max_idx < num_chunks:
            # trim trailing
            spectrogram = F.pad(spectrogram, (0, chunk_size - n_frames))
            spectrograms = spectrogram.expand(num_chunks, *spectrogram.size())
            spectrograms = torch.unbind(spectrograms, dim=0)
            spectrograms = list(spectrograms)
        else:
            min_idx = 0
            spectrograms = []

            for chunk_idx in range(num_chunks):
                max_idx = int(valid_max_idx * (chunk_idx + 1) / num_chunks)

                if self.training:
                    start_idx = torch.randint(min_idx, max_idx, ())
                    start_idx = start_idx.item()
                else:
                    start_idx = (min_idx + max_idx) // 2

                end_idx = start_idx + chunk_size
                _, _spectrogram, _ = torch.split(
                    spectrogram,
                    [start_idx, chunk_size, n_frames - end_idx],
                    dim=-1,
                )
                spectrograms.append(_spectrogram)

                min_idx = max_idx

        return spectrograms

    def _pad_chunks(self, spectrogram: torch.Tensor) -> List[torch.Tensor]:
        chunk_size = self.chunk_size
        num_chunks = self.num_chunks
        pad_mode = self.pad_mode
        n_frames = spectrogram.size(-1)

        if pad_mode == "replicate+constant":
            repeats = chunk_size // n_frames
            repeats = (1,) * (spectrogram.dim() - 1) + (repeats,)
            spectrogram = spectrogram.repeat(*repeats)
        elif pad_mode == "replicate":
            repeats = (chunk_size - 1) // n_frames + 1
            repeats = (1,) * (spectrogram.dim() - 1) + (repeats,)
            spectrogram = spectrogram.repeat(*repeats)
        else:
            raise ValueError(f"Invalid {pad_mode} is found as pad_mode.")

        # trim or pad
        spectrogram = F.pad(spectrogram, (0, chunk_size - spectrogram.size(-1)))
        spectrograms = spectrogram.expand(num_chunks, *spectrogram.size())
        spectrograms = torch.unbind(spectrograms, dim=0)
        spectrograms = list(spectrograms)

        return spectrograms


class LAIONAudioEncoder2023MelSpectrogram(LAIONCLAPAudioEncoder2023MelSpectrogram):
    """Alias of LAIONCLAPAudioEncoder2023MelSpectrogram."""


class LAIONAudioEncoder2023MelSpectrogramFusion(LAIONCLAPAudioEncoder2023MelSpectrogramFusion):
    """Alias of LAIONCLAPAudioEncoder2023MelSpectrogramFusion."""
