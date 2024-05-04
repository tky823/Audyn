import glob
import json
import os
import re
import tarfile
import warnings
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type

import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..composer import Composer
from ..webdataset import (
    supported_audio_extensions,
    supported_json_extensions,
    supported_text_extensions,
    supported_torchdump_extensions,
)
from .distributed import DistributedAudioSetWebDatasetWeightedRandomSampler
from .sampler import AudioSetWebDatasetWeightedRandomSampler

__all__ = [
    "WeightedAudioSetWebDataset",
    "DistributedWeightedAudioSetWebDataset",
    "PaSSTAudioSetWebDataset",
    "DistributedPaSSTAudioSetWebDataset",
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

    .. note::

        After finishing all the processes, it is recommended to close opened files
        by calling ``.close()``.

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
        files: Dict[str, _PicklableFile] = {}

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
        self.worker_id = None

        with open(list_path) as f:
            assert len(self.ytids) == sum(1 for _ in f)

        # set composer and sampler
        self.composer = None
        self.set_sampler(
            feature_dir,
            length,
            replacement=replacement,
            smooth=smooth,
            ytids=self.ytids,
        )

        self.close_all()

    @classmethod
    def instantiate_dataset(
        cls,
        list_path: str,
        feature_dir: str,
        length: int,
        *args,
        replacement: bool = True,
        smooth: float = 1,
        composer: Callable[[Any], Any] = None,
        decode_audio_as_waveform: Optional[bool] = None,
        decode_audio_as_monoral: Optional[bool] = None,
        **kwargs,
    ) -> "WeightedAudioSetWebDataset":
        dataset = cls(
            list_path,
            feature_dir,
            length,
            *args,
            replacement=replacement,
            smooth=smooth,
            **kwargs,
        )

        if composer is None:
            if decode_audio_as_waveform is None:
                decode_audio_as_waveform = True

            if decode_audio_as_monoral is None:
                decode_audio_as_monoral = True

            composer = Composer(
                decode_audio_as_waveform=decode_audio_as_waveform,
                decode_audio_as_monoral=decode_audio_as_monoral,
            )
        else:
            if decode_audio_as_waveform is not None:
                warnings.warn(
                    "decode_audio_as_waveform is given, but ignored.", UserWarning, stacklevel=2
                )

            if decode_audio_as_monoral is not None:
                warnings.warn(
                    "decode_audio_as_monoral is given, but ignored.", UserWarning, stacklevel=2
                )

        dataset = dataset.compose(composer)

        return dataset

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.worker_id is None:
            # should be initialized
            worker_info = get_worker_info()

            if worker_info is None:
                self.worker_id = 0
                num_workers = 1
            else:
                self.worker_id = worker_info.id
                num_workers = worker_info.num_workers

            num_total_samples = self.sampler.num_samples
            num_samples_per_worker = num_total_samples // num_workers

            if self.worker_id < num_total_samples % num_workers:
                num_samples_per_worker += 1

            self.sampler.num_samples = num_samples_per_worker

            for url in self.files.keys():
                self.files[url].close()
                self.files[url] = _PicklableFile(url)

        decoding_iterator = self._decode()

        if self.composer is None:
            yield from decoding_iterator
        else:
            yield from self.composer(decoding_iterator)

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

    def compose(self, composer: Composer) -> "WeightedAudioSetWebDataset":
        """Set composer to dataset.

        Args:
            composer (Composer): Module to process each dict-like sample in a batch.

        Returns:
            WeightedAudioSetWebDataset: Dataset using composer.

        """
        if self.composer is not None:
            raise ValueError("Composer is already defined.")

        self.composer = composer

        return self

    def _decode(self) -> Iterator[Dict[str, Any]]:
        """Return decoding iterator called in __iter__."""
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

                # based on
                # https://github.com/webdataset/webdataset/blob/f11fd66c163722c607ec99475a6f3cb880ec35b8/webdataset/autodecode.py#L156

                if ext in supported_json_extensions:
                    decoded = json.loads(binary)
                elif ext in supported_text_extensions:
                    decoded = binary.decode()
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

    def close_all(self, *args, **kwargs) -> None:
        """Close all tar files."""
        for url in self.files.keys():
            self.files[url].close(*args, **kwargs)


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

    @classmethod
    def instantiate_dataset(
        cls,
        list_path: str,
        feature_dir: str,
        length: int,
        *args,
        replacement: bool = True,
        smooth: float = 1,
        num_workers: int = 0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        composer: Callable[[Any], Any] = None,
        decode_audio_as_waveform: Optional[bool] = None,
        decode_audio_as_monoral: Optional[bool] = None,
        **kwargs,
    ) -> "DistributedWeightedAudioSetWebDataset":
        dataset = cls(
            list_path,
            feature_dir,
            length,
            *args,
            replacement=replacement,
            smooth=smooth,
            num_workers=num_workers,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            drop_last=drop_last,
            **kwargs,
        )

        if composer is None:
            if decode_audio_as_waveform is None:
                decode_audio_as_waveform = True

            if decode_audio_as_monoral is None:
                decode_audio_as_monoral = True

            composer = Composer(
                decode_audio_as_waveform=decode_audio_as_waveform,
                decode_audio_as_monoral=decode_audio_as_monoral,
            )
        else:
            if decode_audio_as_waveform is not None:
                warnings.warn(
                    "decode_audio_as_waveform is given, but ignored.", UserWarning, stacklevel=2
                )

            if decode_audio_as_monoral is not None:
                warnings.warn(
                    "decode_audio_as_monoral is given, but ignored.", UserWarning, stacklevel=2
                )

        dataset = dataset.compose(composer)

        return dataset

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
    ) -> None:
        super().__init__(
            list_path,
            feature_dir,
            length=length,
            replacement=replacement,
            smooth=smooth,
        )

    @classmethod
    def instantiate_dataset(
        cls,
        list_path: str,
        feature_dir: str,
        length: int,
        *args,
        replacement: bool = True,
        smooth: float = 1000,
        composer: Callable[[Any], Any] = None,
        decode_audio_as_waveform: Optional[bool] = None,
        decode_audio_as_monoral: Optional[bool] = None,
        **kwargs,
    ) -> "PaSSTAudioSetWebDataset":
        return super().instantiate_dataset(
            list_path,
            feature_dir,
            length,
            *args,
            replacement=replacement,
            smooth=smooth,
            composer=composer,
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
            **kwargs,
        )


class DistributedPaSSTAudioSetWebDataset(DistributedWeightedAudioSetWebDataset):
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
        smooth: float = 1000,
        num_workers: int = 0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            list_path,
            feature_dir,
            length,
            replacement=replacement,
            smooth=smooth,
            num_workers=num_workers,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            drop_last=drop_last,
        )


class _PicklableFile:
    """Wrapper class of io.BufferedReader to pickle."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.file = open(path, mode="rb")

    def __reduce__(self) -> Tuple[Type, Tuple[str]]:
        self.file.close()
        return self.__class__, (self.path,)

    def seek(self, *args, **kwargs) -> int:
        """Wrapper of file.seek."""
        return self.file.seek(*args, **kwargs)

    def read(self, *args, **kwargs) -> bytes:
        """Wrapper of file.read."""
        return self.file.read(*args, **kwargs)

    def close(self, *args, **kwargs) -> None:
        """Wrapper of file.close."""
        return self.file.close(*args, **kwargs)
