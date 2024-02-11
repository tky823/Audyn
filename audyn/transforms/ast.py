from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kaldi import KaldiMelSpectrogram

__all__ = [
    "SelfSupervisedAudioSpectrogramTransformerMelSpectrogram",
    "SSASTMelSpectrogram",
]


class SelfSupervisedAudioSpectrogramTransformerMelSpectrogram(nn.Module):
    """Mel-spectrogram transform for self-supervised audio spectrogram transformer (SSAST).

    Args:
        sample_rate (int): Sampling rate called as sample_frequency in
            torchaudio.compliance.kaldi.fbank.
        n_mels (int, optional): Number of mel filterbanks called as num_mel_bins in
            torchaudio.compliance.kaldi.fbank.
        n_frames (int, optional): Number of time frames. Shorter spectrogram is padded
            with 0. Longer spectrogram is trimed.
        take_log (bool): Whether to take log features.

    """

    # TODO: Mel-spectrogram transform for AST

    def __init__(
        self,
        sample_rate: int,
        n_mels: Optional[int] = None,
        n_frames: Optional[int] = None,
        take_log: bool = True,
    ) -> None:
        super().__init__()

        self.n_frames = n_frames
        self.take_log = take_log

        fbank_kwargs = {
            "dither": 0.0,
            "frame_shift": 10,
            "htk_compat": True,
            "use_energy": False,
            "window_type": "hanning",
        }

        self.transform = KaldiMelSpectrogram(sample_rate, n_mels=n_mels, fbank_kwargs=fbank_kwargs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Mel-spectrogram transform.

        Args:
            waveform (torch.Tensor): Waveform of shape (batch_size, timesteps)
                or (batch_size, 1, timesteps).

        Returns:
            torch.Tensor: Mel-spectrogram of shape (batch_size, n_mels, n_frames)
                or (batch_size, 1, n_mels, n_frames).

        """
        n_frames = self.n_frames
        take_log = self.take_log

        waveform = waveform - waveform.mean(dim=-1, keepdim=True)
        spectrogram = self.transform(waveform)

        if not take_log:
            spectrogram = torch.exp(spectrogram)

        if n_frames is not None:
            padding = n_frames - spectrogram.size(-1)
            spectrogram = F.pad(spectrogram, (0, padding))

        return spectrogram


class SSASTMelSpectrogram(SelfSupervisedAudioSpectrogramTransformerMelSpectrogram):
    """Alias of SelfSupervisedAudioSpectrogramTransformerMelSpectrogram."""
