from typing import List, Optional

import torch.distributed as dist

from .sampler import RandomStemsMUSDB18Sampler


class DistributedRandomStemsMUSDB18Sampler(RandomStemsMUSDB18Sampler):
    """RandomStemsMUSDB18Sampler for DDP.

    Args:
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    """

    def __init__(
        self,
        track_names: List[str],
        replacement: bool = True,
        num_samples: Optional[int] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = False,
        generator=None,
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
        self.drop_last = drop_last

        num_samples_per_replica = num_samples // num_replicas

        if num_samples % num_replicas > 0 and not drop_last:
            num_samples_per_replica += 1

        super().__init__(
            track_names,
            replacement=replacement,
            num_samples=num_samples_per_replica,
            generator=generator,
        )
