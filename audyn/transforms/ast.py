from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as aT

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
        freq_mask_param (int, optional): Parameter of frequency masking. This masking is
            deactivated when ``self.training`` is ``False``.
        time_mask_param (int, optional): Parameter of time masking. This masking is
            deactivated when ``self.training`` is ``False``.
        mean (float): Mean of spectrogram in dataset. Default: ``0``.
        std (float): Standard deviation of spectrogram in dataset. Default: ``1``.
        take_log (bool): Whether to take log features.

    """

    # ported from
    # https://github.com/YuanGongND/ssast/blob/a1a3eecb94731e226308a6812f2fbf268d789caf/src/dataloader.py#L95-L144
    # and
    # https://github.com/YuanGongND/ssast/blob/a1a3eecb94731e226308a6812f2fbf268d789caf/src/dataloader.py#L183-L202
    # TODO: Mel-spectrogram transform for AST

    def __init__(
        self,
        sample_rate: int,
        n_mels: Optional[int] = None,
        n_frames: Optional[int] = None,
        freq_mask_param: Optional[int] = None,
        time_mask_param: Optional[int] = None,
        mean: float = 0,
        std: float = 1,
        take_log: bool = True,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.mean = mean
        self.std = std
        self.take_log = take_log

        fbank_kwargs = {
            "dither": 0.0,
            "frame_shift": 10,
            "htk_compat": True,
            "use_energy": False,
            "window_type": "hanning",
        }

        self.transform = KaldiMelSpectrogram(sample_rate, n_mels=n_mels, fbank_kwargs=fbank_kwargs)

        if freq_mask_param is None:
            self.frequency_masking = None
        else:
            self.frequency_masking = aT.FrequencyMasking(freq_mask_param)

        if time_mask_param is None:
            self.time_masking = None
        else:
            self.time_masking = aT.TimeMasking(time_mask_param)

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
        mean = self.mean
        std = self.std
        take_log = self.take_log

        waveform = waveform - waveform.mean(dim=-1, keepdim=True)
        spectrogram = self.transform(waveform)

        if n_frames is not None:
            padding = n_frames - spectrogram.size(-1)
            spectrogram = F.pad(spectrogram, (0, padding))

        if self.training:
            if self.frequency_masking is not None:
                spectrogram = self.frequency_masking(spectrogram)

            if self.time_masking is not None:
                spectrogram = self.time_masking(spectrogram)

        spectrogram = (spectrogram - mean) / (2 * std)

        if not take_log:
            spectrogram = torch.exp(spectrogram)

        return spectrogram


class SSASTMelSpectrogram(SelfSupervisedAudioSpectrogramTransformerMelSpectrogram):
    """Alias of SelfSupervisedAudioSpectrogramTransformerMelSpectrogram."""
