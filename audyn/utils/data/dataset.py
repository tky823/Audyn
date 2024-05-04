import glob
import os
import warnings
from typing import Any, Callable, Dict, Optional

import torch
import webdataset as wds
from torch.utils.data import Dataset

from .composer import Composer

__all__ = [
    "TorchObjectDataset",
    "SortableTorchObjectDataset",
    "WebDatasetWrapper",
]

available_dump_formats = ["torch", "webdataset"]


class TorchObjectDataset(Dataset):
    """Dataset for .pth objects.

    Args:
        list_path (str): Path to list file containing .pth filenames.
        feature_dir (str): Path to directory containing .pth objects.

    """

    def __init__(self, list_path: str, feature_dir: str) -> None:
        super().__init__()

        self.feature_dir = feature_dir
        self.filenames = []

        with open(list_path) as f:
            for line in f:
                self.filenames.append(line.strip("\n"))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename = self.filenames[idx]
        feature_path = os.path.join(self.feature_dir, f"{filename}.pth")
        data = torch.load(feature_path, map_location=lambda storage, loc: storage)

        return data

    def __len__(self) -> int:
        return len(self.filenames)


class SortableTorchObjectDataset(TorchObjectDataset):
    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        sort_by_length: bool = True,
        sort_key: str = None,
        length_dim: int = -1,
    ) -> None:
        """Dataset for .pth objects sorted by a certain feature.

        .. note::

            If tensor of ``sort_key`` is 0-D (i.e. scalar),
            the value itself is treated as length.

        Args:
            list_path (str): Path to list file containing .pth filenames.
            feature_dir (str): Path to directory containing .pth objects.
            sort_by_length (bool): If ``True``, objects are sorted.
            sort_key (str): Key to sort objects.
            length_dim (int): Dimension to sort.

        """
        if sort_key is None:
            raise ValueError("Specify sort_key.")

        super().__init__(list_path=list_path, feature_dir=feature_dir)

        if sort_by_length:
            lengths = {}

            for filename in self.filenames:
                feature_path = os.path.join(self.feature_dir, f"{filename}.pth")
                data = torch.load(feature_path, map_location=lambda storage, loc: storage)

                if data[sort_key].dim() == 0:
                    lengths[filename] = data[sort_key].item()
                else:
                    lengths[filename] = data[sort_key].size(length_dim)

            # longest is first
            lengths = sorted(lengths.items(), key=lambda x: x[1], reverse=True)
            self.filenames = [filename for filename, _ in lengths]


class WebDatasetWrapper(wds.WebDataset):
    """Wrapper class of WebDataset to call ``with_epoch``, ``with_length``,
    and ``decode`` (and ``shuffle`` if necessary) for instantiation.

    ``WebDatasetWrapper.instantiate_dataset`` is typically called for instantiation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def instantiate_dataset(
        cls,
        list_path: str,
        feature_dir: str,
        *args,
        detshuffle: bool = True,
        shuffle_size: Any = None,
        nodesplitter: Callable = wds.split_by_node,
        composer: Callable[[Any], Any] = None,
        decode_audio_as_waveform: Optional[bool] = None,
        decode_audio_as_monoral: Optional[bool] = None,
        **kwargs,
    ) -> "WebDatasetWrapper":
        """Instantiate WebDatasetWrapper.

        Args:
            args: Positional arguments given to WebDataset.
            kwargs: Keyword arguments given to WebDataset.
            shuffle_size (any, optional): Shuffle size for training dataset.
            nodesplitter (callable): Module to split dataset by node.
            decode_audio_as_waveform (bool, optional): If ``True``, audio is decoded as waveform
                tensor and sampling rate is ignored. Otherwise, audio is decoded as tuple of
                waveform tensor and sampling rate. This parameter is given to Composer class.
                When composer is specified, this parameter is not used. Default: ``True``.
            decode_audio_as_monoral (bool, optional): If ``True``, decoded audio is treated as
                monoral waveform of shape (num_samples,) by reducing channel dimension. Otherwise,
                shape of waveform is (num_channels, num_samples), which is returned by
                ``torchaudio.load``. When composer is specified, this parameter is not used.
                Default: ``True``.

        Returns:
            WebDatasetWrapper: Wrapper of WebDataset. ``with_epoch``, ``with_length``,
                ``shuffle``, and ``decode`` are called if necessary.

        """
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

        template_path = os.path.join(feature_dir, "*.tar")
        urls = []

        for url in sorted(glob.glob(template_path)):
            urls.append(url)

        with open(list_path) as f:
            length = sum(1 for _ in f)

        dataset = cls(
            urls,
            feature_dir,
            *args,
            detshuffle=detshuffle,
            nodesplitter=nodesplitter,
            **kwargs,
        )
        dataset = dataset.with_epoch(length).with_length(length)

        if shuffle_size is not None:
            if not detshuffle:
                warnings.warn(
                    "detshuffle=True is highly recommended for training "
                    "in terms of reproducibility."
                )

            dataset = dataset.shuffle(shuffle_size)

        dataset = dataset.decode()
        dataset = dataset.compose(composer)

        return dataset
