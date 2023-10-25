from typing import Dict, Iterable, Iterator, List

import torch
from packaging import version
from torch.utils.data.sampler import Sampler

__all__ = ["SequentialBatchSampler", "DynamicBatchSampler"]


IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class SequentialBatchSampler(Sampler):
    """Batch sampler that returns sequential block indices.

    Examples:

        >>> sampler = SequentialBatchSampler(range(10), batch_size=3, seed=0, drop_last=True)
        >>> for indices in sampler:
        ...     print(indices)
        ...
        [6, 7, 8]
        [0, 1, 2]
        [3, 4, 5]
        >>> sampler = SequentialBatchSampler(range(10), batch_size=3, seed=0, drop_last=False)
        >>> for indices in sampler:
        ...     print(indices)
        ...
        [0, 1, 2]
        [3, 4, 5]
        [9]
        [6, 7, 8]

    """

    def __init__(
        self,
        data_source: Iterable[Dict[str, torch.Tensor]],
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if IS_TORCH_LT_2_1:
            super().__init__(data_source)
        else:
            super().__init__()

        self.data_source = data_source

        self.epoch = 0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch_size = self.batch_size

        assert len(self.data_source) > 0, "self.data_source is empty."

        num_batches = len(self.data_source) // batch_size

        if not self.drop_last and len(self.data_source) % batch_size > 0:
            num_batches += 1

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_indices = torch.randperm(num_batches, generator=g)
        else:
            batch_indices = list(range(num_batches))

        indices = list(range(len(self.data_source)))

        for idx in batch_indices:
            yield indices[batch_size * idx : batch_size * (idx + 1)]

    def __len__(self) -> int:
        batch_size = self.batch_size
        num_batches = len(self.data_source) // batch_size

        if not self.drop_last and len(self.data_source) % batch_size > 0:
            num_batches += 1

        return num_batches

    def set_epoch(self, epoch: int) -> None:
        """Set epoch to ensure consistency with resuming training.

        Args:
            epoch (int): Epoch number.

        """
        self.epoch = epoch


class DynamicBatchSampler(Sampler):
    """Batch sampler where each batch size is dynamically determined by the length of samples.

    When ``__init__`` is called, this class iteratively accesses samples of the given data source
    until the accumulated length exceeds ``batch_length``.
    """

    def __init__(
        self,
        data_source: Iterable[Dict[str, torch.Tensor]],
        key: str,
        batch_length: int = 0,
        dim: int = -1,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if IS_TORCH_LT_2_1:
            super().__init__(data_source)
        else:
            super().__init__()

        self.batched_indices = []
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

        cum_length = 0
        cum_indices = []

        for idx in range(len(data_source)):
            data = data_source[idx]
            feature = data[key]

            if isinstance(feature, torch.Tensor):
                length = feature.size(dim)
            elif isinstance(feature, (list, str)):
                length = len(feature)
            else:
                # TODO: Generic type, 0-dim input
                raise NotImplementedError(f"Type {type(feature)} is not supported.")

            cum_length += length
            cum_indices.append(idx)

            if cum_length >= batch_length:
                self.batched_indices.append(cum_indices)
                cum_length = 0
                cum_indices = []

        if len(cum_indices) > 0 and not drop_last:
            self.batched_indices.append(cum_indices)

    def __iter__(self) -> Iterator[List[int]]:
        assert len(self.batched_indices) > 0, "self.batched_indices is empty."

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            for idx in torch.randperm(len(self.batched_indices), generator=g):
                yield self.batched_indices[idx]
        else:
            for indices in self.batched_indices:
                yield indices

    def __len__(self) -> int:
        return len(self.batched_indices)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch to ensure consistency with resuming training.

        Args:
            epoch (int): Epoch number.

        """
        self.epoch = epoch
