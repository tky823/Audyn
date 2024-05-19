from typing import Any, Dict

import torch

from ...composer import Composer
from . import primary_labels


class BirdCLEF2024PrimaryLabelComposer(Composer):
    """Composer to include primary label of BirdCLEF2024.

    Args:
        label_name_key (str): Key of prmary label name in given sample.
        label_index_key (str): Key of prmary label index to add to given sample.
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
        label_name_key: str,
        label_index_key: str,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.label_name_key = label_name_key
        self.label_index_key = label_index_key

        self.primary_labels = primary_labels

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        label_name_key = self.label_name_key
        label_index_key = self.label_index_key

        sample = super().process(sample)

        label_name = sample[label_name_key]
        label_index = self.primary_labels.index(label_name)
        sample[label_index_key] = torch.full((), fill_value=label_index, dtype=torch.long)

        return sample
