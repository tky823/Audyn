from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torchaudio.functional as aF

from ....transforms.clap import (
    LAIONCLAPAudioEncoder2023MelSpectrogram,
    LAIONCLAPAudioEncoder2023MelSpectrogramFusion,
)
from ..composer import Composer

__all__ = [
    "LAIONCLAPAudioEncoder2023Composer",
    "LAIONAudioEncoder2023Composer",
]


class LAIONCLAPAudioEncoder2023Composer(Composer):
    """Composer for LAIONCLAPAudioEncoder2023.

    Args:
        melspectrogram_transform (LAIONCLAPAudioEncoder2023MelSpectrogram or nn.Module):
            Module to transform waveform to Mel-spectrogram.
        fusion_transform (LAIONCLAPAudioEncoder2023MelSpectrogramFusion or nn.Module):
            Module to fuse Mel-spectrogram.
        waveform_key (str): Key of waveform in given sample.
        sample_rate_key (str): Key of sampling rate in given sample.
        melspectrogram_key (str): Key of Mel-spectrogram to add to given sample.
        training (bool): If ``training=True``, ``melspectrogram_transform.train()`` and
            ``fusion_transform.train()`` are called. Otherwise, ``.eval()`` is called.

    """

    def __init__(
        self,
        melspectrogram_transform: Union[LAIONCLAPAudioEncoder2023MelSpectrogram, nn.Module],
        fusion_transform: Union[LAIONCLAPAudioEncoder2023MelSpectrogramFusion, nn.Module],
        waveform_key: str = "waveform",
        sample_rate_key: str = "sample_rate",
        melspectrogram_key: str = "melspectrogram",
        fused_melspectrogram_key: str = "fused_melspectrogram",
        training: bool = False,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.melspectrogram_transform = melspectrogram_transform
        self.fusion_transform = fusion_transform

        self.waveform_key = waveform_key
        self.sample_rate_key = sample_rate_key
        self.melspectrogram_key = melspectrogram_key
        self.fused_melspectrogram_key = fused_melspectrogram_key

        self.training = training

        if training:
            if hasattr(self.melspectrogram_transform, "train") and callable(
                self.melspectrogram_transform.train
            ):
                self.melspectrogram_transform.train()

            if hasattr(self.fusion_transform, "train") and callable(self.fusion_transform.train):
                self.melspectrogram_transform.train()
        else:
            if hasattr(self.melspectrogram_transform, "eval") and callable(
                self.melspectrogram_transform.eval
            ):
                self.melspectrogram_transform.eval()

            if hasattr(self.fusion_transform, "eval") and callable(self.fusion_transform.eval):
                self.melspectrogram_transform.eval()

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        waveform_key = self.waveform_key
        sample_rate_key = self.sample_rate_key
        melspectrogram_key = self.melspectrogram_key
        fused_melspectrogram_key = self.fused_melspectrogram_key

        target_sample_rate = self.melspectrogram_transform.sample_rate
        waveform = sample[waveform_key]
        sample_rate = sample[sample_rate_key]

        if isinstance(sample_rate, torch.Tensor):
            sample_rate = sample_rate.item()

        if sample_rate != target_sample_rate:
            waveform = aF.resample(waveform, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate

        melspectrogram = self.melspectrogram_transform(waveform)
        fused_melspectrogram = self.fusion_transform(melspectrogram)

        sample_rate = torch.tensor(sample_rate, dtype=torch.long)

        output = {
            waveform_key: waveform,
            sample_rate_key: sample_rate,
            melspectrogram_key: melspectrogram,
            fused_melspectrogram_key: fused_melspectrogram,
        }

        return output


class LAIONAudioEncoder2023Composer(LAIONCLAPAudioEncoder2023Composer):
    """Alias of LAIONCLAPAudioEncoder2023Composer."""
