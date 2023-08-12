import math
from typing import Iterator, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from .sampler import DynamicBatchSampler


class DistributedSequentialBatchSampler:
    r"""Sampler that restricts data loading to a subset of the dataset.

    The implementation is based on
    https://github.com/pytorch/pytorch/blob/43e71cddb0dc85b43a98238740bd5f8584d841fd/torch/utils/data/distributed.py#L13-L136

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )

        self.dataset = dataset
        self.batch_size = batch_size

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        self.batched_indices = []

        indices = list(range(len(dataset)))

        for idx in range(len(dataset) // batch_size):
            self.batched_indices.append(indices[batch_size * idx : batch_size * (idx + 1)])

        num_drops = len(dataset) % batch_size

        if not self.drop_last and num_drops > 0:
            self.batched_indices.append(indices[-num_drops:])

        num_batch_drops = len(self.batched_indices) % self.num_replicas

        if not self.drop_last and num_batch_drops > 0:
            self.num_samples = math.ceil(len(self.batched_indices) / self.num_replicas)
        else:
            self.num_samples = (len(self.batched_indices) - num_batch_drops) // self.num_replicas

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.batched_indices), generator=g).tolist()
        else:
            indices = list(range(len(self.batched_indices)))

        if self.drop_last:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        else:
            # TODO: add test when self.drop_last = False
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            repeated_indices = indices * math.ceil(padding_size / len(indices))
            indices += repeated_indices[:padding_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.total_size // self.num_replicas

        for idx in indices:
            yield self.batched_indices[idx]

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch to ensure consistency with other replicas.

        Args:
            epoch (int): Epoch number.

        """
        self.epoch = epoch


class DistributedDynamicBatchSampler(DynamicBatchSampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    The implementation is based on
    https://github.com/pytorch/pytorch/blob/43e71cddb0dc85b43a98238740bd5f8584d841fd/torch/utils/data/distributed.py#L13-L136

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    """

    def __init__(
        self,
        dataset: Dataset,
        key: str,
        batch_length: int = 0,
        dim: int = -1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset,
            key,
            batch_length=batch_length,
            dim=dim,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        num_drops = len(self.batched_indices) % self.num_replicas

        if not self.drop_last and num_drops > 0:
            self.num_samples = math.ceil(len(self.batched_indices) / self.num_replicas)
        else:
            self.num_samples = (len(self.batched_indices) - num_drops) // self.num_replicas

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.batched_indices), generator=g).tolist()
        else:
            indices = list(range(len(self.batched_indices)))

        if self.drop_last:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        else:
            # TODO: add test when self.drop_last = False
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            repeated_indices = indices * math.ceil(padding_size / len(indices))
            indices += repeated_indices[:padding_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.total_size // self.num_replicas

        for idx in indices:
            yield self.batched_indices[idx]

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch to ensure consistency with other replicas.

        Args:
            epoch (int): Epoch number.

        """
        self.epoch = epoch
