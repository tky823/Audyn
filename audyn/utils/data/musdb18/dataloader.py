from typing import Any, Callable, Iterable, List, Optional, Union

from torch.utils.data import DataLoader, Dataset

from .dataset import MUSDB18


class RandomMixMUSDB18DataLoader(DataLoader):

    def __init__(
        self,
        dataset: MUSDB18,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        sampler: Optional[Union[Sampler, Iterable]] = None,
        batch_sampler: Optional[Union[Sampler[List], Iterable[List]]] = None,
        num_workers: int = 0,
        collate_fn: Callable[[List], Any] | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ) -> None:
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )
