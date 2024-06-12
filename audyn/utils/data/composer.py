import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch

from .webdataset import decode_audio, supported_audio_extensions

__all__ = [
    "Composer",
    "AudioFeatureExtractionComposer",
    "SequentialComposer",
    "LogarithmTaker",
]


class Composer:
    """Composer given to process each sample in list of samples.

    This class is mainly used for webdataset, but is also useful for torch dataset.

    .. note::

        To include additional processing, please implement ``process`` method.

    """

    def __init__(
        self,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        self.decode_audio_as_waveform = decode_audio_as_waveform
        self.decode_audio_as_monoral = decode_audio_as_monoral

    def decode(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        from . import rename_webdataset_keys

        for key in sample.keys():
            # ported from
            # https://github.com/webdataset/webdataset/blob/f11fd66c163722c607ec99475a6f3cb880ec35b8/webdataset/autodecode.py#L418-L434
            ext = re.sub(r".*[.]", "", key)

            if ext in supported_audio_extensions:
                data = sample[key]

                if isinstance(data, bytes):
                    sample[key] = decode_audio(
                        data,
                        ext,
                        decode_audio_as_monoral=self.decode_audio_as_monoral,
                        decode_audio_as_waveform=self.decode_audio_as_waveform,
                    )

        sample = rename_webdataset_keys(sample)

        return sample

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process to edit each sample."""
        return sample

    def __call__(self, samples: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        for sample in samples:
            sample = self.decode(sample)
            sample = self.process(sample)

            yield sample


class AudioFeatureExtractionComposer(Composer):
    """Composer for feature extraction from audio.

    Args:
        feature_extractor (callable): Feature extractor that takes waveform (torch.Tensor).
            Returned feature is saved as ``feature_key`` in a dictionary.
        audio_key (str): Key of audio to extract feature.
        feature_key (str): Key of feature.
        training (bool): If ``True``, ``feature_extractor.train()`` is called. Otherwise,
            ``feature_extractor.eval()`` is called (only if possible).

    """

    def __init__(
        self,
        feature_extractor: Callable[[torch.Tensor], Any],
        audio_key: str,
        feature_key: str,
        training: bool = False,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.feature_extractor = feature_extractor
        self.audio_key = audio_key
        self.feature_key = feature_key
        self.training = training

        if training:
            assert hasattr(
                self.feature_extractor, "train"
            ), "self.feature_extractor should have train."
            assert callable(
                self.feature_extractor.train
            ), "self.feature_extractor.train should be callable."

            self.feature_extractor.train()
        else:
            if hasattr(self.feature_extractor, "eval") and callable(self.feature_extractor.eval):
                self.feature_extractor.eval()

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        audio_key = self.audio_key
        feature_key = self.feature_key

        sample = super().process(sample)

        audio = sample[audio_key]
        feature = self.feature_extractor(audio)
        sample[feature_key] = feature

        return sample


class SequentialComposer(Composer):
    """Module to apply multiple composers."""

    def __init__(
        self,
        *composers,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.composers: List[Composer] = list(composers)

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process to edit each sample."""
        for composer in self.composers:
            sample = composer.process(sample)

        return sample


class LogarithmTaker(Composer):
    """Composer to take log.

    Args:
        input_key (str): Key of tensor to apply log.
        output_key (str): Key of tensor to store log feature.
        flooring (callable or float): Flooring function or value to avoid zero division.

    """

    def __init__(
        self,
        input_key: str,
        output_key: str,
        flooring: Optional[Union[Callable[[torch.Tensor], torch.Tensor], float]] = None,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.input_key = input_key
        self.output_key = output_key
        self.flooring = flooring

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_key = self.input_key
        output_key = self.output_key

        sample = super().process(sample)
        feature = sample[input_key]

        if self.flooring is not None:
            if isinstance(self.flooring):
                feature = torch.clamp(feature, min=self.flooring)
            else:
                feature = self.flooring(feature)

        sample[output_key] = torch.log(feature)

        return sample
