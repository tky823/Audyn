import glob
import json
import os
import re
import tarfile
from io import BufferedReader, BytesIO
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF
import torchaudio.transforms as aT
from torch.utils.data import IterableDataset, get_worker_info

from ..dataset import Composer
from ..webdataset import (
    supported_audio_extensions,
    supported_json_extensions,
    supported_text_extensions,
    supported_torchdump_extensions,
)
from . import num_tags as num_audioset_tags
from . import tags as audioset_tags
from .distributed import DistributedAudioSetWebDatasetWeightedRandomSampler
from .sampler import AudioSetWebDatasetWeightedRandomSampler

__all__ = [
    "WeightedAudioSetWebDataset",
    "DistributedWeightedAudioSetWebDataset",
    "AudioSetMultiLabelComposer",
    "PaSSTAudioSetMultiLabelComposer",
]


class WeightedAudioSetWebDataset(IterableDataset):
    """AudioSet using WebDataset with weighted random sampling.

    The implementation is based on one described in [#koutini2022efficient]_.
    In this dataset, samples with rare tags are more likely to be taken.

    Args:
        list_path (str): Path to list file containing filenames.
        feature_dir (str): Path to directory containing .tar files.
        length (int): Number of samples at each epoch.
        replacement (bool): If ``True``, samples are taken with replacement.
        smooth (int): Offset to frequency of each class. In [#koutini2022efficient]_, ``1000``
            is used. Default: ``1``.

    .. [#koutini2022efficient]
        K. Koutini et al., "Efficient training of audio transformers with patchout,"
        in *Interspeech*, 2022.

    """

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        length: int,
        replacement: bool = True,
        smooth: float = 1,
    ) -> None:
        super().__init__()

        ytids = set()
        mapping = {}
        files: Dict[str, BufferedReader] = {}

        for url in sorted(glob.glob(os.path.join(feature_dir, "*.tar"))):
            with tarfile.open(url) as f:
                for tarinfo in f:
                    ytid, key = tarinfo.name.split(".", maxsplit=1)

                    if ytid not in ytids:
                        ytids.add(ytid)
                        mapping[ytid] = {
                            "__url__": url,
                            "data": {},
                        }

                    data = {
                        "offset_data": tarinfo.offset_data,
                        "size": tarinfo.size,
                    }
                    mapping[ytid]["data"][key] = data

            files[url] = _PicklableFile(url)

        self.ytids = sorted(list(mapping.keys()))
        self.mapping = mapping
        self.files = files

        with open(list_path) as f:
            assert len(self.ytids) == sum(1 for _ in f)

        self.set_sampler(
            feature_dir,
            length,
            replacement=replacement,
            smooth=smooth,
            ytids=self.ytids,
        )

    def __iter__(self) -> Iterator:
        worker_info = get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            num_total_samples = self.sampler.num_samples

            num_samples_per_worker = num_total_samples // num_workers

            if worker_id < num_total_samples % num_workers:
                num_samples_per_worker += 1

            self.sampler.num_samples = num_samples_per_worker

        for index in self.sampler:
            ytid = self.ytids[index]
            mapping = self.mapping[ytid]
            url = mapping["__url__"]
            data: Dict[str, Any] = mapping["data"]
            f = self.files[url]

            sample = {
                "__key__": ytid,
                "__url__": url,
            }

            for key, value in data.items():
                if key.startswith("__"):
                    continue

                offset_data = value["offset_data"]
                size = value["size"]

                f.seek(offset_data)
                binary = f.read(size)
                ext = re.sub(r".*[.]", "", key)

                if ext in supported_json_extensions:
                    binary = binary.decode("utf-8")
                    decoded = json.loads(binary)
                elif ext in supported_text_extensions:
                    decoded = binary.decode("utf-8")
                elif ext in supported_torchdump_extensions:
                    binary = BytesIO(binary)
                    decoded = torch.load(binary)
                elif ext in supported_audio_extensions:
                    # NOTE: Decoding is applied in composer like ordinary webdataset.
                    decoded = binary
                else:
                    raise ValueError(f"Invalid key {key} is detected.")

                sample[key] = decoded

            yield sample

    def __len__(self) -> int:
        return self.sampler.num_samples

    def set_sampler(
        self,
        feature_dir: str,
        length: int,
        replacement: bool = True,
        smooth: float = 1,
        ytids: List[str] = None,
    ) -> None:
        if ytids is None:
            ytids = self.ytids

        self.sampler = AudioSetWebDatasetWeightedRandomSampler(
            feature_dir,
            length,
            replacement=replacement,
            smooth=smooth,
            ytids=ytids,
        )


class DistributedWeightedAudioSetWebDataset(WeightedAudioSetWebDataset):
    """AudioSet using WebDataset with weighted random sampling for distributed training.

    See ``audyn.utils.data.audioset.dataset.WeightedAudioSetWebDataset`` for the
    details of arguments.

    """

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        length: int,
        replacement: bool = True,
        smooth: float = 1,
        num_workers: int = 0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.num_workers = num_workers
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last

        super().__init__(
            list_path,
            feature_dir,
            length=length,
            replacement=replacement,
            smooth=smooth,
        )

    def set_sampler(
        self,
        feature_dir: str,
        length: int,
        replacement: bool = True,
        smooth: float = 1,
        ytids: List[str] = None,
    ) -> None:
        if ytids is None:
            ytids = self.ytids

        self.sampler = DistributedAudioSetWebDatasetWeightedRandomSampler(
            feature_dir,
            length,
            num_replicas=self.num_replicas,
            rank=self.rank,
            seed=self.seed,
            drop_last=self.drop_last,
            replacement=replacement,
            smooth=smooth,
            ytids=ytids,
        )


class PaSSTAudioSetWebDataset(WeightedAudioSetWebDataset):
    """AudioSet using WebDataset with weighted random sampling.

    The implementation is based on one described in [#koutini2022efficient]_.
    In this dataset, samples with rare tags are more likely to be taken.

    Args:
        list_path (str): Path to list file containing filenames.
        feature_dir (str): Path to directory containing .tar files.
        length (int): Number of samples at each epoch.
        replacement (bool): If ``True``, samples are taken with replacement.
        smooth (int): Offset to frequency of each class. Default: ``1000``, which is
            different from ``WeightedAudioSetWebDataset``.

    """

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        length: int,
        replacement: bool = True,
        smooth: float = 1000,
        decode_audio_as_monoral: bool = True,
        decode_audio_as_waveform: bool = True,
        generator=None,
    ) -> None:
        super().__init__(
            list_path,
            feature_dir,
            length=length,
            replacement=replacement,
            smooth=smooth,
            decode_audio_as_monoral=decode_audio_as_monoral,
            decode_audio_as_waveform=decode_audio_as_waveform,
            generator=generator,
        )


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


class PaSSTAudioSetMultiLabelComposer(AudioSetMultiLabelComposer):
    """Composer to include multi-label of AudioSet for PaSST.

    This class returns sample containing ``waveform``, ``melspectrogram``, ``filename``,
    ``tags``, and ``multilabel``.

    Args:
        dump_format: Dump format. Now, only ``webdataset`` is supported.
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
        dump_format: str,
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

        self.dump_format = dump_format
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


class _PicklableFile:
    """Wrapper class of io.BufferedReader to pickle."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.file = open(path, mode="rb")

    def __reduce__(self) -> tuple[Type, Tuple[str]]:
        self.file.close()
        return self.__class__, (self.path,)

    def seek(self, *args, **kwargs) -> Any:
        """Wrapper of file.seek."""
        return self.file.seek(*args, **kwargs)

    def read(self, *args, **kwargs) -> Any:
        """Wrapper of file.read."""
        return self.file.read(*args, **kwargs)

    def close(self, *args, **kwargs) -> Any:
        """Wrapper of file.close."""
        return self.file.close(*args, **kwargs)
