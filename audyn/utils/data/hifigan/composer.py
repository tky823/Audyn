from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torchaudio.functional as aF

from ....transforms.hifigan import HiFiGANMelSpectrogram
from ....transforms.slicer import WaveformSlicer
from ..composer import Composer

__all__ = ["HiFiGANComposer"]


class HiFiGANComposer(Composer):
    """Composer for HiFiGAN.

    Args:
        melspectrogram_transform (HiFiGANMelSpectrogram or nn.Module): Module to transform waveform
            to Mel-spectrogram.
        slicer (WaveformSlicer or nn.Module): Wavefor slicer.
        waveform_key (str): Key of waveform in given sample.
        sample_rate_key (str): Key of sampling rate in given sample.
        melspectrogram_key (str): Key of Mel-spectrogram to add to given sample.
        waveform_slice_key (str): Key of sliced waveform to add to given sample.
        melspectrogram_slice_key (str): Key of sliced Mel-spectrogram to add to given sample.
        training (bool): If ``training=True``, ``melspectrogram_transform.train()`` and
            ``slicer.train()`` are called. Otherwise, ``.eval()`` is called.

    """

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
            if hasattr(self.slicer, "train") and callable(self.slicer.train):
                self.slicer.train()

            if hasattr(self.melspectrogram_transform, "train") and callable(
                self.melspectrogram_transform.train
            ):
                self.melspectrogram_transform.train()
        else:
            if hasattr(self.slicer, "eval") and callable(self.slicer.eval):
                self.slicer.eval()

            if hasattr(self.melspectrogram_transform, "eval") and callable(
                self.melspectrogram_transform.eval
            ):
                self.melspectrogram_transform.eval()

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        waveform_key = self.waveform_key
        sample_rate_key = self.sample_rate_key
        melspectrogram_key = self.melspectrogram_key
        waveform_slice_key = self.waveform_slice_key
        melspectrogram_slice_key = self.melspectrogram_slice_key

        target_sample_rate = self.melspectrogram_transform.sample_rate
        waveform = sample[waveform_key]
        sample_rate = sample[sample_rate_key]

        if isinstance(sample_rate, torch.Tensor):
            sample_rate = sample_rate.item()

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

        sample_rate = torch.tensor(sample_rate, dtype=torch.long)

        output = {
            waveform_key: waveform,
            sample_rate_key: sample_rate,
            melspectrogram_key: melspectrogram,
            waveform_slice_key: waveform_slice,
            melspectrogram_slice_key: melspectrogram_slice,
        }

        return output
