import glob
import json
import os
import re
import tarfile
from io import BufferedReader, BytesIO
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

import torch
from torch.utils.data import IterableDataset, get_worker_info

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
                    binary = binary.decode()
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


class _PicklableFile:
    """Wrapper class of io.BufferedReader to pickle."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.file = open(path, mode="rb")

    def __reduce__(self) -> Tuple[Type, Tuple[str]]:
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
