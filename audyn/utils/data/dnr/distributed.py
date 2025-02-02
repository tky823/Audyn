from typing import Iterator, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info

from .sampler import RandomStemsDNRSampler

__all__ = [
    "DistributedRandomStemsDNRSampler",
]


class DistributedRandomStemsDNRSampler(RandomStemsDNRSampler):
    """DnR sampler to generate mixture composed by randomly selected tracks.

    Args:
        track_names (list): Track name list.
        num_samples (int, optional): Number of sampler per epoch at each rank.
            ``len(track_names) // num_replicas`` is used by default.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    """

    def __init__(
        self,
        track_names: List[str],
        num_samples: Optional[int] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
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

        if num_samples is None:
            num_samples = len(track_names) // num_replicas

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        super().__init__(
            track_names,
            num_samples=num_samples,
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
