from typing import Optional, Union

import torch.nn as nn
import torchaudio.transforms as aT

from .._common.composer import BirdCLEFPrimaryLabelComposer
from . import primary_labels as birdclef2021_primary_labels

__all__ = [
    "BirdCLEF2021PrimaryLabelComposer",
]


class BirdCLEF2021PrimaryLabelComposer(BirdCLEFPrimaryLabelComposer):
    """Composer to include primary label of BirdCLEF2021.

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
            melspectrogram_transform,
            birdclef2021_primary_labels,
            audio_key=audio_key,
            sample_rate_key=sample_rate_key,
            label_name_key=label_name_key,
            filename_key=filename_key,
            waveform_key=waveform_key,
            melspectrogram_key=melspectrogram_key,
            label_index_key=label_index_key,
            sample_rate=sample_rate,
            duration=duration,
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
            training=training,
        )
