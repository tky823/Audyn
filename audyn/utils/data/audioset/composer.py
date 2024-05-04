from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF
import torchaudio.transforms as aT

from ..composer import Composer
from . import num_tags as num_audioset_tags
from . import tags as audioset_tags

__all__ = [
    "AudioSetMultiLabelComposer",
    "ASTAudioSetMultiLabelComposer",
]


class AudioSetMultiLabelComposer(Composer):
    """Composer to include multi-label of AudioSet.

    Args:
        tags_key (str): Key of tags in given sample.
        multilabel_key (str): Key of multi-label to add to given sample.
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
        tags_key: str,
        multilabel_key: str,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.tags_key = tags_key
        self.multilabel_key = multilabel_key

        tag_to_index = {}

        for idx, tag in enumerate(audioset_tags):
            _tag = tag["tag"]
            tag_to_index[_tag] = idx

        self.tag_to_index = tag_to_index

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        tags_key = self.tags_key
        multilabel_key = self.multilabel_key

        sample = super().process(sample)

        tags = sample[tags_key]
        labels = torch.zeros((num_audioset_tags,))

        for tag in tags:
            tag_idx = self.tag_to_index[tag]
            labels[tag_idx] = 1

        sample[multilabel_key] = labels

        return sample


class ASTAudioSetMultiLabelComposer(AudioSetMultiLabelComposer):
    """Composer to include multi-label of AudioSet for AST.

    This class returns sample containing ``waveform``, ``melspectrogram``, ``filename``,
    ``tags``, and ``multilabel``.

    Args:
        melspectrogram_transform (torchaudio.transforms.MelSpectrogram or nn.Module):
            Module to transform waveform into Mel-spectrogram.
        audio_key (str): Key of ``audio`` (without extension) saved in tar files.
        sample_rate_key (str): Key of ``sample_rate`` (without extension) saved in tar files.
        tags_key (str): Key of ``sample_rate`` (without extension) saved in tar files.
        filename_key (str): Key of ``filename`` (without extension) saved in tar files.
        waveform_key (str): Key of ``waveform`` to save in sample.
        melspectrogram_key (str): Key of ``melspectrogram`` to save in sample.
        multilabel_key (str): Key of ``multilabel`` to save in sample.
        duration (float): Duration of waveform. Default: ``10``.
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
        audio_key: str = None,
        sample_rate_key: str = None,
        tags_key: str = None,
        filename_key: str = "filename",
        waveform_key: str = "waveform",
        melspectrogram_key: str = "melspectrogram",
        multilabel_key: str = None,
        duration: float = 10,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            tags_key,
            multilabel_key,
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.melspectrogram_transform = melspectrogram_transform
        self.audio_key = audio_key
        self.sample_rate_key = sample_rate_key
        self.filename_key = filename_key
        self.waveform_key = waveform_key
        self.melspectrogram_key = melspectrogram_key
        self.duration = duration

        if audio_key is None:
            raise ValueError("audio_key is required.")

        if sample_rate_key is None:
            raise ValueError("sample_rate_key is required.")

        if tags_key is None:
            raise ValueError("tags_key is required.")

        if hasattr(melspectrogram_transform, "sample_rate"):
            self.melspectrogram_sample_rate: int = melspectrogram_transform.sample_rate
        else:
            raise NotImplementedError("Sampling rate of melspectrogram_transform should be set.")

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        audio_key = self.audio_key
        sample_rate_key = self.sample_rate_key
        tags_key = self.tags_key
        filename_key = self.filename_key
        waveform_key = self.waveform_key
        melspectrogram_key = self.melspectrogram_key
        multilabel_key = self.multilabel_key
        duration = self.duration
        melspectrogram_sample_rate = self.melspectrogram_sample_rate

        sample = super().process(sample)
        length = int(melspectrogram_sample_rate * duration)

        waveform = sample[audio_key]
        sample_rate = sample[sample_rate_key]
        tags = sample[tags_key]
        multilabel = sample[multilabel_key]
        filename = sample[filename_key]

        if sample_rate != melspectrogram_sample_rate:
            waveform = aF.resample(waveform, sample_rate, melspectrogram_sample_rate)

        if length is not None:
            padding = length - waveform.size(-1)
            waveform = F.pad(waveform, (0, padding))

        melspectrogram = self.melspectrogram_transform(waveform)

        output = {
            waveform_key: waveform,
            melspectrogram_key: melspectrogram,
            tags_key: tags,
            multilabel_key: multilabel,
            filename_key: filename,
        }

        return output
