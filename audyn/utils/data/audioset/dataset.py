import glob
import json
import os
import re
import tarfile
from io import BufferedReader, BytesIO
from typing import Any, Dict, Iterator

import torch
from torch.utils.data import IterableDataset

from ..webdataset import (
    decode_audio,
    supported_audio_extensions,
    supported_json_extensions,
    supported_text_extensions,
    supported_torchdump_extensions,
)
from .sampler import AudioSetWebDatasetWeightedRandomSampler

__all__ = [
    "WeightedAudioSetWebDataset",
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
        generator=None,
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

            files[url] = open(url, mode="rb")

        self.ytids = sorted(list(mapping.keys()))
        self.mapping = mapping
        self.files = files

        with open(list_path) as f:
            assert len(self.ytids) == sum(1 for _ in f)

        self.sampler = AudioSetWebDatasetWeightedRandomSampler(
            feature_dir,
            length,
            replacement=replacement,
            smooth=smooth,
            ytids=self.ytids,
            generator=generator,
        )

    def __iter__(self) -> Iterator:
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
                    decoded = decode_audio(binary, ext)
                else:
                    raise ValueError(f"Invalid key {key} is detected.")

                sample[key] = decoded

            yield sample


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
        generator=None,
    ) -> None:
        super().__init__(
            list_path,
            feature_dir,
            length=length,
            replacement=replacement,
            smooth=smooth,
            generator=generator,
        )
