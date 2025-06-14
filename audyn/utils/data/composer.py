import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
import torch.nn.functional as F
import torchaudio.functional as aF

from .webdataset import decode_audio, supported_audio_extensions

__all__ = [
    "Composer",
    "AudioFeatureExtractionComposer",
    "SequentialComposer",
    "LogarithmTaker",
    "SynchronousWaveformSlicer",
    "LabelToOnehot",
    "LabelsToMultihot",
    "ResamplingComposer",
    "UnpackingAudioComposer",
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
            if isinstance(self.flooring, float):
                feature = torch.clamp(feature, min=self.flooring)
            else:
                feature = self.flooring(feature)

        sample[output_key] = torch.log(feature)

        return sample


class SynchronousWaveformSlicer(Composer):
    """Composer to slice multiple waveforms synchronously by fixed length or duration.

    Args:
        input_keys (list): Keys of tensor to slice waveforms.
        output_keys (list): Keys of tensor to store sliced waveforms.
        length (int, optional): Length of waveform slice.
        duration (float, optional): Duration of waveform slice.
        sample_rate (int, optional): Sampling rate of waveform.
        seed (int): Random seed.
        training (bool): If ``True``, waveforms are sliced at random. Default: ``False``.

    """

    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        length: int = None,
        duration: float = None,
        sample_rate: int = None,
        seed: int = 0,
        training: bool = False,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        assert len(input_keys) == len(output_keys)

        if length is None:
            assert duration is not None and sample_rate is not None

            length = int(sample_rate * duration)
        else:
            assert duration is None and sample_rate is None

        self.input_keys = input_keys
        self.output_keys = output_keys
        self.training = training

        self.length = length
        self.duration = duration
        self.sample_rate = sample_rate
        self.generator = torch.Generator()

        self.generator.manual_seed(seed)

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_keys = self.input_keys
        output_keys = self.output_keys
        g = self.generator

        sample = super().process(sample)

        # use first input key
        input_key = input_keys[0]
        waveform = sample[input_key]
        length = self.length
        orig_length = waveform.size(-1)
        padding = length - orig_length

        if self.training:
            if padding > 0:
                padding_left = torch.randint(0, padding, (), generator=g).item()
                padding_right = padding - padding_left

                for input_key, output_key in zip(input_keys, output_keys):
                    waveform = sample[input_key]
                    waveform_slice = F.pad(waveform, (padding_left, padding_right))
                    sample[output_key] = waveform_slice
            elif padding < 0:
                trimming = -padding
                start_idx = torch.randint(0, trimming, (), generator=g).item()
                end_idx = start_idx + length

                for input_key, output_key in zip(input_keys, output_keys):
                    waveform = sample[input_key]
                    _, waveform_slice, _ = torch.split(
                        waveform, [start_idx, length, orig_length - end_idx], dim=-1
                    )
                    sample[output_key] = waveform_slice
            else:
                for input_key, output_key in zip(input_keys, output_keys):
                    sample[output_key] = sample[input_key].clone()
        else:
            if padding > 0:
                padding_left = padding // 2
                padding_right = padding - padding_left

                for input_key, output_key in zip(input_keys, output_keys):
                    waveform = sample[input_key]
                    waveform_slice = F.pad(waveform, (padding_left, padding_right))
                    sample[output_key] = waveform_slice
            elif padding < 0:
                trimming = -padding
                start_idx = trimming // 2
                end_idx = start_idx + length

                for input_key, output_key in zip(input_keys, output_keys):
                    waveform = sample[input_key]
                    _, waveform_slice, _ = torch.split(
                        waveform, [start_idx, length, orig_length - end_idx], dim=-1
                    )
                    sample[output_key] = waveform_slice
            else:
                for input_key, output_key in zip(input_keys, output_keys):
                    sample[output_key] = sample[input_key].clone()

        return sample


class Rescaler(Composer):
    """Rescale waveform by maximum absolute value.

    Args:
        input_key (str or list): Input key(s) to rescale.
        output_key (str or list): Output key(s) to store rescaled waveform.
        scale (float): Target scale. Default: ``1``.
        eps (float): Tiny value to avoid zero division.
        decode_audio_as_monoral (bool): Whether to decode audio as monoral. Default: ``True``.

    """

    def __init__(
        self,
        input_key: Union[str, List[str]],
        output_key: Union[str, List[str]],
        scale: float = 1,
        eps: float = 1e-8,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.input_key = input_key
        self.output_key = output_key
        self.scale = scale
        self.eps = eps

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_key = self.input_key
        output_key = self.output_key
        scale = self.scale

        sample = super().process(sample)

        if isinstance(input_key, str):
            input_keys = [input_key]
        else:
            input_keys = input_key

        if isinstance(output_key, str):
            output_keys = [output_key]
        else:
            output_keys = output_key

        assert len(output_keys) == len(input_keys)

        for input_key, output_key in zip(input_keys, output_keys):
            waveform = sample[input_key]
            amplitude = torch.abs(waveform)
            max_amplitude = torch.max(amplitude)
            max_amplitude = torch.clamp(max_amplitude, min=self.eps)
            sample[output_key] = scale * (waveform / max_amplitude)

        return sample


class Mixer(Composer):
    """Mixer for training of source separation.

    Args:
        input_keys (list): Input keys to mix.
        output_key (str): Output key to store mixture.
        decode_audio_as_monoral (bool): Whether to decode audio as monoral. Default: ``True``.

    """

    def __init__(
        self,
        input_keys: List[str],
        output_key: str,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.input_keys = input_keys
        self.output_key = output_key

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_keys = self.input_keys
        output_key = self.output_key

        sample = super().process(sample)

        mixture = 0

        for input_key in input_keys:
            mixture = mixture + sample[input_key]

        sample[output_key] = mixture

        return sample


class RescaledMixer(Composer):
    """Mixer for training of source separation.

    Args:
        input_keys (list): Input keys to mix.
        output_key (str): Output key to store mixture.
        scale (float): Target scale. Default: ``1``.
        eps (float): Tiny value to avoid zero division.
        decode_audio_as_monoral (bool): Whether to decode audio as monoral. Default: ``True``.

    Unlike ``audyn.utils.data.Mixer``, amplitude is rescaled by maximum absolute value.
    This operation may change entries of ``input_keys``.

    """

    def __init__(
        self,
        input_keys: List[str],
        output_key: str,
        scale: float = 1,
        eps: float = 1e-8,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.input_keys = input_keys
        self.output_key = output_key
        self.scale = scale
        self.eps = eps

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_keys = self.input_keys
        output_key = self.output_key
        scale = self.scale

        sample = super().process(sample)

        mixture = 0

        for input_key in input_keys:
            mixture = mixture + sample[input_key]

        amplitude = torch.abs(mixture)
        max_amplitude = torch.max(amplitude)
        max_amplitude = torch.clamp(max_amplitude, min=self.eps)
        sample[output_key] = mixture / max_amplitude

        for input_key in input_keys:
            sample[input_key] = scale * (sample[input_key] / max_amplitude)

        return sample


class Stacker(Composer):
    """Stack for training of source separation.

    Args:
        input_keys (list): Input keys to concatenate.
        output_key (str): Output key to store concatenated features.
        dim (int): Dimension to stack.

    """

    def __init__(
        self,
        input_keys: List[str],
        output_key: str,
        dim: int,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.input_keys = input_keys
        self.output_key = output_key
        self.dim = dim

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_keys = self.input_keys
        output_key = self.output_key
        dim = self.dim

        sample = super().process(sample)

        output = []

        for input_key in input_keys:
            output.append(sample[input_key])

        sample[output_key] = torch.stack(output, dim=dim)

        return sample


class LabelToOnehot(Composer):
    """Composer for classification task.

    Args:
        label_key (str): Key of label in classification.
        feature_key (str): Key to store one-hot feature.
        labels (list): List of labels.

    """

    def __init__(
        self,
        label_key: str,
        feature_key: str,
        labels: List[str],
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.label_key = label_key
        self.feature_key = feature_key
        self.labels = labels

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        label_key = self.label_key
        feature_key = self.feature_key
        labels = self.labels
        num_labels = len(labels)

        sample = super().process(sample)

        label = sample[label_key]
        onehot = torch.zeros((num_labels,))
        index = labels.index(label)
        onehot[index] = 1
        sample[feature_key] = onehot

        return sample


class LabelsToMultihot(Composer):
    """Composer for multi-label classification.

    Args:
        labels_key (str): Key of labels in multi-label classification.
        feature_key (str): Key to store multi-hot feature.
        labels (list): List of labels.

    """

    def __init__(
        self,
        label_key: str,
        feature_key: str,
        labels: List[str],
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.label_key = label_key
        self.feature_key = feature_key
        self.labels = labels

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        label_key = self.label_key
        feature_key = self.feature_key
        labels = self.labels
        num_labels = len(labels)

        sample = super().process(sample)

        multihot = torch.zeros((num_labels,))

        for label in sample[label_key]:
            index = labels.index(label)
            multihot[index] = 1

        sample[feature_key] = multihot

        return sample


class ResamplingComposer(Composer):
    """Composer for multi-label classification.

    Args:
        new_freq (int): Target sampling rate.
        audio_key (str): Key of audio in given sample.
        sample_rate_key (str): Key of sampling rate in given sample.

    """

    def __init__(
        self,
        new_freq: int,
        audio_key: str,
        sample_rate_key: str,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.new_freq = new_freq
        self.audio_key = audio_key
        self.sample_rate_key = sample_rate_key

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        new_freq = self.new_freq
        audio_key = self.audio_key
        sample_rate_key = self.sample_rate_key

        sample = super().process(sample)

        waveform = sample[audio_key]
        sample_rate = sample[sample_rate_key]

        if sample_rate != new_freq:
            waveform = aF.resample(waveform, sample_rate, new_freq)

        sample[audio_key] = waveform
        sample[sample_rate_key] = torch.tensor(new_freq, dtype=torch.long)

        return sample


class UnpackingAudioComposer(Composer):
    """Composer to unpack tuple of (waveform, sample_rate) to separate keys.

    Args:
        audio_key (str): Key of audio in given sample.
        waveform_key (str): Key of waveform to store.
        sample_rate_key (str): Key of sampling rate to store.

    """

    def __init__(
        self,
        audio_key: str,
        waveform_key: str,
        sample_rate_key: str,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        assert decode_audio_as_waveform, "decode_audio_as_waveform should be True."

        super().__init__(
            decode_audio_as_waveform=False,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.audio_key = audio_key
        self.waveform_key = waveform_key
        self.sample_rate_key = sample_rate_key

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        audio_key = self.audio_key
        waveform_key = self.waveform_key
        sample_rate_key = self.sample_rate_key
        decode_audio_as_waveform = self.decode_audio_as_waveform

        assert not decode_audio_as_waveform, "decode_audio_as_waveform should be False."

        sample = super().process(sample)

        audio = sample[audio_key]

        assert isinstance(audio, tuple)
        assert (
            len(audio) == 2
        ), f"Audio with {audio_key} should be a tuple of (waveform, sample_rate)."

        waveform, sample_rate = audio

        sample[waveform_key] = waveform
        sample[sample_rate_key] = sample_rate

        return sample
