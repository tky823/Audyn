from typing import Iterator, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info

from .sampler import AudioSetWebDatasetWeightedRandomSampler

__all__ = [
    "DistributedAudioSetWebDatasetWeightedRandomSampler",
]


class DistributedAudioSetWebDatasetWeightedRandomSampler(AudioSetWebDatasetWeightedRandomSampler):
    """Distributed sampler for AudioSetWebDatasetWeightedRandomSampler.

    Args:
        feature_dir (str): Path to directory containing .tar files.
        num_samples (int): Number of samples at each epoch.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        replacement (bool): If ``True``, samples are taken with replacement.
        smooth (int): Offset to frequency of each class. In [#koutini2022efficient]_, ``1000``
            is used. Default: ``1``.
        ytids (list, optional): YouTube IDs. This list is useful to align order of samples between
            sampler and other modules. If ``None``, order of ytids are determined by
            alphabetical order using built-in ``sorted`` function.

    """

    def __init__(
        self,
        feature_dir: str,
        num_samples: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        replacement: bool = True,
        smooth: float = 1,
        ytids: Optional[List[str]] = None,
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

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last

        num_samples_per_replica = num_samples // num_replicas

        if num_samples % num_replicas > 0 and not drop_last:
            num_samples_per_replica += 1

        super().__init__(
            feature_dir,
            num_samples=num_samples_per_replica,
            replacement=replacement,
            smooth=smooth,
            ytids=ytids,
        )

    def __iter__(self) -> Iterator[int]:
        rank = self.rank

        if self.generator is None:
            self.generator = torch.Generator()

            worker_info = get_worker_info()

            if worker_info is None:
                worker_id = 0
                num_workers = 1
            else:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers

            self.generator.manual_seed(self.seed + rank * num_workers + worker_id)

        return super().__iter__()
