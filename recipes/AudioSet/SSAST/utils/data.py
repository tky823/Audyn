from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF
import torchaudio.transforms as aT

from audyn.transforms.ast import SelfSupervisedAudioSpectrogramTransformerMelSpectrogram
from audyn.utils.data.collater import BaseCollater


class SSASTAudioSetCollater(BaseCollater):
    """Collater for SSAST using AudioSet."""

    def __init__(
        self,
        melspectrogram_transform: Union[
            SelfSupervisedAudioSpectrogramTransformerMelSpectrogram,
            aT.MelSpectrogram,
            nn.Module,
        ],
        waveform_key: str = "waveform",
        melspectrogram_key: str = "melspectrogram",
        filename_key: str = "filename",
        duration: float = 10,
    ) -> None:
        super().__init__()

        self.melspectrogram_transform = melspectrogram_transform
        self.waveform_key = waveform_key
        self.melspectrogram_key = melspectrogram_key
        self.filename_key = filename_key
        self.duration = duration

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        waveform_key = self.waveform_key
        melspectrogram_key = self.melspectrogram_key
        filename_key = self.filename_key
        melspectrogram_transform = self.melspectrogram_transform
        duration = self.duration

        if hasattr(melspectrogram_transform, "sample_rate"):
            melspectrogram_sample_rate = melspectrogram_transform.sample_rate
            length = int(melspectrogram_sample_rate * duration)
        else:
            melspectrogram_sample_rate = None
            length = None

            raise NotImplementedError("Sampling rate should be specified now.")

        batched_waveform = []
        batched_melspectrogram = []
        batched_filename = []

        for sample in batch:
            waveform = sample["audio.m4a"]
            sample_rate = sample["sample_rate.pth"]
            filename = sample["__key__"]

            if (
                melspectrogram_sample_rate is not None
                and sample_rate != melspectrogram_sample_rate
            ):
                waveform = aF.resample(waveform, sample_rate, melspectrogram_sample_rate)

            padding = length - waveform.size(-1)
            waveform = F.pad(waveform, (0, padding))
            melspectrogram = self.melspectrogram_transform(waveform)

            batched_waveform.append(waveform)
            batched_melspectrogram.append(melspectrogram)
            batched_filename.append(filename)

        batched_waveform = torch.stack(batched_waveform, dim=0)
        batched_melspectrogram = torch.stack(batched_melspectrogram, dim=0)

        output = {
            waveform_key: batched_waveform,
            melspectrogram_key: batched_melspectrogram,
            filename_key: batched_filename,
        }

        return output
