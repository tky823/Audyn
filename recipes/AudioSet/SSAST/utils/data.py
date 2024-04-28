from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF
import torchaudio.transforms as aT

from audyn.transforms.ast import SelfSupervisedAudioSpectrogramTransformerMelSpectrogram
from audyn.utils.data.collater import BaseCollater


class SSASTAudioSetCollater(BaseCollater):
    """Collater for SSAST using AudioSet.

    Args:
        duration (float, optional): Duration to trim or pad.

    """

    def __init__(
        self,
        dump_format: str,
        melspectrogram_transform: Union[
            SelfSupervisedAudioSpectrogramTransformerMelSpectrogram,
            aT.MelSpectrogram,
            nn.Module,
        ],
        audio_key_in: str = "audio.m4a",
        sample_rate_key_in: str = "sample_rate.pth",
        filename_key_in: str = "filename",
        waveform_key_out: str = "waveform",
        melspectrogram_key_out: str = "melspectrogram",
        filename_key_out: str = "filename",
        duration: Optional[float] = 10,
    ) -> None:
        super().__init__()

        self.dump_format = dump_format
        self.melspectrogram_transform = melspectrogram_transform
        self.audio_key_in = audio_key_in
        self.sample_rate_key_in = sample_rate_key_in
        self.filename_key_in = filename_key_in
        self.waveform_key_out = waveform_key_out
        self.melspectrogram_key_out = melspectrogram_key_out
        self.filename_key_out = filename_key_out
        self.duration = duration

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.dump_format != "webdataset":
            raise ValueError("Only webdataset is supported as dump_format.")

        audio_key_in = self.audio_key_in
        sample_rate_key_in = self.sample_rate_key_in
        filename_key_in = self.filename_key_in
        waveform_key_out = self.waveform_key_out
        melspectrogram_key_out = self.melspectrogram_key_out
        filename_key_out = self.filename_key_out
        melspectrogram_transform = self.melspectrogram_transform
        duration = self.duration

        if hasattr(melspectrogram_transform, "sample_rate"):
            melspectrogram_sample_rate: int = melspectrogram_transform.sample_rate
        else:
            raise NotImplementedError("Sampling rate of melspectrogram_transform should be set.")

        if duration is None:
            length = None
            max_length = 0
        else:
            length = int(melspectrogram_sample_rate * duration)
            max_length = length

        batched_waveform = []
        batched_melspectrogram = []
        batched_filename = []

        for sample in batch:
            waveform = sample[audio_key_in]
            sample_rate = sample[sample_rate_key_in]
            filename = sample[filename_key_in]

            if sample_rate != melspectrogram_sample_rate:
                waveform = aF.resample(waveform, sample_rate, melspectrogram_sample_rate)

            if length is not None:
                padding = length - waveform.size(-1)
                waveform = F.pad(waveform, (0, padding))

            # NOTE: When length is not None, max_length cannot be updated.
            max_length = max(max_length, waveform.size(-1))

            batched_waveform.append(waveform)
            batched_filename.append(filename)

        for waveform in batched_waveform:
            # NOTE: When length is not None, padding is 0.
            padding = max_length - waveform.size(-1)
            waveform = F.pad(waveform, (0, padding))

            melspectrogram = self.melspectrogram_transform(waveform)
            batched_melspectrogram.append(melspectrogram)

        batched_waveform = torch.stack(batched_waveform, dim=0)
        batched_melspectrogram = torch.stack(batched_melspectrogram, dim=0)

        output = {
            waveform_key_out: batched_waveform,
            melspectrogram_key_out: batched_melspectrogram,
            filename_key_out: batched_filename,
        }

        return output
