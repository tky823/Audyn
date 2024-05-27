from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF
import torchaudio.transforms as aT

from ...composer import Composer
from . import primary_labels as birdclef2024_primary_labels

__all__ = [
    "BirdCLEF2024PrimaryLabelComposer",
    "BirdCLEF2024AudioComposer",
]


class BirdCLEF2024PrimaryLabelComposer(Composer):
    """Composer to include primary label of BirdCLEF2024.

    Args:
        audio_key (str): Key of audio.
        sample_rate_key (str): Key of sampling rate.
        label_name_key (str): Key of prmary label name in given sample.
        filename_key (str): Key of filename in given sample.
        waveform_key (str): Key of waveform to add to given sample.
        melspectrogram_key (str): Key of Mel-spectrogram to add to given sample.
        label_index_key (str): Key of prmary label index to add to given sample.
        sample_rate (int): Target sampling rate. Default: ``32000``.
        duration (float, optional): Duration of audio to trim or pad. Default: ``15``.
        decode_audio_as_waveform (bool): If ``True``, audio is decoded as waveform
            tensor and sampling rate is ignored. Otherwise, audio is decoded as tuple of
            waveform tensor and sampling rate. This parameter is given to Composer class.
            When composer is specified, this parameter is not used. Default: ``True``.
        decode_audio_as_monoral (bool): If ``True``, decoded audio is treated as
            monoral waveform of shape (num_samples,) by reducing channel dimension. Otherwise,
            shape of waveform is (num_channels, num_samples), which is returned by
            ``torchaudio.load``. When composer is specified, this parameter is not used.
            Default: ``True``.

    """

    def __init__(
        self,
        melspectrogram_transform: Union[
            aT.MelSpectrogram,
            nn.Module,
        ],
        audio_key: str,
        sample_rate_key: str,
        label_name_key: str,
        filename_key: str = "filename",
        waveform_key: str = "waveform",
        melspectrogram_key: str = "melspectrogram",
        label_index_key: str = "label_index",
        sample_rate: int = 32000,
        duration: Optional[float] = 15,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
        training: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.melspectrogram_transform = melspectrogram_transform

        self.audio_key = audio_key
        self.sample_rate_key = sample_rate_key
        self.label_name_key = label_name_key
        self.filename_key = filename_key
        self.waveform_key = waveform_key
        self.melspectrogram_key = melspectrogram_key
        self.label_index_key = label_index_key

        self.primary_labels = birdclef2024_primary_labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.training = training

        assert hasattr(self.melspectrogram_transform, "train")
        assert callable(self.melspectrogram_transform.train)
        assert hasattr(self.melspectrogram_transform, "eval")
        assert callable(self.melspectrogram_transform.eval)

        if self.training:
            self.melspectrogram_transform.train()
        else:
            self.melspectrogram_transform.eval()

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        audio_key = self.audio_key
        sample_rate_key = self.sample_rate_key
        label_name_key = self.label_name_key
        filename_key = self.filename_key
        waveform_key = self.waveform_key
        melspectrogram_key = self.melspectrogram_key
        label_index_key = self.label_index_key
        target_sample_rate = self.sample_rate
        duration = self.duration

        sample = super().process(sample)

        audio = sample[audio_key]
        sample_rate = sample[sample_rate_key]
        sample_rate_dtype = sample[sample_rate_key].dtype
        sample_rate = sample_rate.item()

        assert isinstance(audio, torch.Tensor), f"{type(audio)} is not supported."

        if sample_rate != target_sample_rate:
            audio = aF.resample(audio, sample_rate, target_sample_rate)
            sample[sample_rate_key] = torch.full(
                (), fill_value=sample_rate, dtype=sample_rate_dtype
            )

        if duration is not None:
            length = int(target_sample_rate * duration)
            padding = length - audio.size(-1)

            if padding > 0:
                if self.training:
                    padding_left = torch.randint(0, padding, ()).item()
                else:
                    padding_left = padding // 2

                padding_right = padding - padding_left
            elif padding < 0:
                padding = -padding

                if self.training:
                    padding_left = torch.randint(0, padding, ()).item()
                else:
                    padding_left = padding // 2

                padding_right = padding - padding_left
                padding_left = -padding_left
                padding_right = -padding_right
            else:
                padding_left = 0
                padding_right = 0

            audio = F.pad(audio, (padding_left, padding_right))

        label_name = sample[label_name_key]
        label_index = self.primary_labels.index(label_name)
        label_index = torch.full((), fill_value=label_index, dtype=torch.long)

        melspectrogram = self.melspectrogram_transform(audio)

        output = {
            waveform_key: audio,
            melspectrogram_key: melspectrogram,
            label_index_key: label_index,
            filename_key: sample[filename_key],
        }

        return output


class BirdCLEF2024AudioComposer(Composer):
    """Composer to include audio of BirdCLEF2024.

    Args:
        audio_key (str): Key of audio.
        sample_rate_key (str): Key of sampling rate.
        filename_key (str): Key of filename in given sample.
        waveform_key (str): Key of waveform to add to given sample.
        melspectrogram_key (str): Key of Mel-spectrogram to add to given sample.
        sample_rate (int): Target sampling rate. Default: ``32000``.
        duration (float, optional): Duration of audio to trim or pad. Default: ``15``.
        decode_audio_as_waveform (bool): If ``True``, audio is decoded as waveform
            tensor and sampling rate is ignored. Otherwise, audio is decoded as tuple of
            waveform tensor and sampling rate. This parameter is given to Composer class.
            When composer is specified, this parameter is not used. Default: ``True``.
        decode_audio_as_monoral (bool): If ``True``, decoded audio is treated as
            monoral waveform of shape (num_samples,) by reducing channel dimension. Otherwise,
            shape of waveform is (num_channels, num_samples), which is returned by
            ``torchaudio.load``. When composer is specified, this parameter is not used.
            Default: ``True``.

    """

    def __init__(
        self,
        melspectrogram_transform: Union[
            aT.MelSpectrogram,
            nn.Module,
        ],
        audio_key: str,
        sample_rate_key: str,
        filename_key: str = "filename",
        melspectrogram_key: str = "melspectrogram",
        waveform_key: str = "waveform",
        sample_rate: int = 32000,
        duration: Optional[float] = 15,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
        training: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.melspectrogram_transform = melspectrogram_transform

        self.audio_key = audio_key
        self.sample_rate_key = sample_rate_key
        self.filename_key = filename_key
        self.waveform_key = waveform_key
        self.melspectrogram_key = melspectrogram_key

        self.primary_labels = birdclef2024_primary_labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.training = training

        assert hasattr(self.melspectrogram_transform, "train")
        assert callable(self.melspectrogram_transform.train)
        assert hasattr(self.melspectrogram_transform, "eval")
        assert callable(self.melspectrogram_transform.eval)

        if self.training:
            self.melspectrogram_transform.train()
        else:
            self.melspectrogram_transform.eval()

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        audio_key = self.audio_key
        sample_rate_key = self.sample_rate_key
        filename_key = self.filename_key
        waveform_key = self.waveform_key
        melspectrogram_key = self.melspectrogram_key
        target_sample_rate = self.sample_rate
        duration = self.duration

        sample = super().process(sample)

        audio = sample[audio_key]
        sample_rate = sample[sample_rate_key]
        sample_rate_dtype = sample[sample_rate_key].dtype
        sample_rate = sample_rate.item()

        assert isinstance(audio, torch.Tensor), f"{type(audio)} is not supported."

        if sample_rate != target_sample_rate:
            audio = aF.resample(audio, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
            sample[sample_rate_key] = torch.full(
                (), fill_value=sample_rate, dtype=sample_rate_dtype
            )

        if duration is not None:
            length = int(sample_rate * duration)
            padding = length - audio.size(-1)

            if padding > 0:
                if self.training:
                    padding_left = torch.randint(0, padding, ()).item()
                else:
                    padding_left = padding // 2

                padding_right = padding - padding_left
            elif padding < 0:
                padding = -padding

                if self.training:
                    padding_left = torch.randint(0, padding, ()).item()
                else:
                    padding_left = padding // 2

                padding_right = padding - padding_left
                padding_left = -padding_left
                padding_right = -padding_right
            else:
                padding_left = 0
                padding_right = 0

            audio = F.pad(audio, (padding_left, padding_right))

        melspectrogram = self.melspectrogram_transform(audio)

        output = {
            waveform_key: audio,
            melspectrogram_key: melspectrogram,
            sample_rate_key: sample[sample_rate_key],
            filename_key: sample[filename_key],
        }

        return output
