from typing import Any, Dict, Optional

import torch
from packaging import version
from torch.utils.data.dataloader import DataLoader, _collate_fn_t, _worker_init_fn_t
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

from .distributed import DistributedDynamicBatchSampler, DistributedSequentialBatchSampler
from .sampler import DynamicBatchSampler, SequentialBatchSampler

__all__ = [
    "SequentialBatchDataLoader",
    "DistributedDataLoader",
    "DistributedSequentialBatchDataLoader",
    "DynamicBatchDataLoader",
    "DistributedDynamicBatchDataLoader",
]


class SequentialBatchDataLoader(DataLoader):
    """Data loader where each batch is extracted by sequential block indices.

    This class uses SequentialBatchSampler as a batch sampler.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        seed: int = 0,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        persistent_workers: bool = False,
        **kwargs,
    ) -> None:
        batch_sampler = SequentialBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last
        )

        self._validate_kwargs(kwargs)

        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            persistent_workers=persistent_workers,
            **kwargs,
        )

    @staticmethod
    def _validate_kwargs(kwargs) -> None:
        """Validate given keyword arguments."""
        valid_keys = {"prefetch_factor"}

        if version.parse(torch.__version__) >= version.parse("1.12"):
            valid_keys.add("pin_memory_device")

        invalid_keys = set(kwargs.keys()) - valid_keys

        assert invalid_keys == set(), f"Invalid keys {invalid_keys} are given."


class DistributedDataLoader(DataLoader):
    """Data loader for distributed data parallel.

    This class uses DistributedSampler as a batch sampler.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: Optional[bool] = None,
        seed: int = 0,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        persistent_workers: bool = False,
        **kwargs,
    ) -> None:
        sampler = DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        _validate_dataloader_kwargs(kwargs)

        super().__init__(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            persistent_workers=persistent_workers,
            **kwargs,
        )


class DistributedSequentialBatchDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: Optional[bool] = None,
        seed: int = 0,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        *,
        persistent_workers: bool = False,
        **kwargs,
    ) -> None:
        batch_sampler = DistributedSequentialBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        _validate_dataloader_kwargs(kwargs)

        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            persistent_workers=persistent_workers,
            **kwargs,
        )


class DynamicBatchDataLoader(DataLoader):
    """Data loader where each batch size is determined dynamically.

    This class uses DynamicBatchSampler as a batch sampler.
    """

    def __init__(
        self,
        dataset: Dataset,
        key: str,
        batch_length: int = 0,
        dim: int = -1,
        shuffle: Optional[bool] = None,
        seed: int = 0,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        *,
        persistent_workers: bool = False,
        **kwargs,
    ) -> None:
        batch_sampler = DynamicBatchSampler(
            dataset,
            key=key,
            batch_length=batch_length,
            dim=dim,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        _validate_dataloader_kwargs(kwargs)

        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            persistent_workers=persistent_workers,
            **kwargs,
        )


class DistributedDynamicBatchDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        key: str,
        batch_length: int = 0,
        dim: int = -1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: Optional[bool] = None,
        seed: int = 0,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        *,
        persistent_workers: bool = False,
        **kwargs,
    ) -> None:
        batch_sampler = DistributedDynamicBatchSampler(
            dataset,
            key=key,
            batch_length=batch_length,
            dim=dim,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        _validate_dataloader_kwargs(kwargs)

        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            persistent_workers=persistent_workers,
            **kwargs,
        )


def _validate_dataloader_kwargs(kwargs: Dict[str, Any]) -> None:
    """Validate given keyword arguments."""
    valid_keys = {"prefetch_factor"}

    if version.parse(torch.__version__) >= version.parse("1.12"):
        valid_keys.add("pin_memory_device")

    invalid_keys = set(kwargs.keys()) - valid_keys

    assert invalid_keys == set(), f"Invalid keys {invalid_keys} are given."
