from typing import Dict, Union

import torch
import torch.nn as nn
import torchaudio.functional as aF

from ....transforms.hifigan import HiFiGANMelSpectrogram
from ....transforms.slicer import WaveformSlicer
from ..composer import Composer

__all__ = ["HiFiGANComposer"]


class HiFiGANComposer(Composer):
    """Composer for HiFiGAN."""

    def __init__(
        self,
        melspectrogram_transform: Union[HiFiGANMelSpectrogram, nn.Module],
        slicer: Union[WaveformSlicer, nn.Module],
        waveform_key: str = "waveform",
        sample_rate_key: str = "sample_rate",
        melspectrogram_key: str = "melspectrogram",
        waveform_slice_key: str = "waveform_slice",
        melspectrogram_slice_key: str = "melspectrogram_slice",
        training: bool = False,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.melspectrogram_transform = melspectrogram_transform
        self.slicer = slicer

        self.waveform_key = waveform_key
        self.sample_rate_key = sample_rate_key
        self.melspectrogram_key = melspectrogram_key
        self.waveform_slice_key = waveform_slice_key
        self.melspectrogram_slice_key = melspectrogram_slice_key

        self.training = training

        if training:
            self.slicer.train()
            self.melspectrogram_transform.train()
        else:
            self.slicer.eval()
            self.melspectrogram_transform.eval()

    def process(self, sample: Dict[str, torch.Any]) -> Dict[str, torch.Any]:
        waveform_key = self.waveform_key
        sample_rate_key = self.sample_rate_key
        melspectrogram_key = self.melspectrogram_key
        waveform_slice_key = self.waveform_slice_key
        melspectrogram_slice_key = self.melspectrogram_slice_key

        target_sample_rate = self.melspectrogram_transform.sample_rate
        waveform = sample[waveform_key]
        sample_rate: int = sample[sample_rate_key].item()

        if waveform.dim() == 1:
            # HiFi-GAN requires 3D target as batch.
            waveform = waveform.unsqueeze(dim=0)

        if sample_rate != target_sample_rate:
            waveform = aF.resample(waveform, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate

        melspectrogram = self.melspectrogram_transform(waveform)
        waveform_slice = self.slicer(waveform)
        melspectrogram_slice = self.melspectrogram_transform(waveform_slice)

        # HiFi-GAN requires 3D input as batch.
        if melspectrogram.dim() == 3:
            melspectrogram = melspectrogram.squeeze(dim=0)

        if melspectrogram_slice.dim() == 3:
            melspectrogram_slice = melspectrogram_slice.squeeze(dim=0)

        sample[waveform_key] = waveform
        sample[sample_rate_key] = torch.tensor(sample_rate, dtype=torch.long)
        sample[melspectrogram_key] = melspectrogram
        sample[waveform_slice_key] = waveform_slice
        sample[melspectrogram_slice_key] = melspectrogram_slice

        return sample
