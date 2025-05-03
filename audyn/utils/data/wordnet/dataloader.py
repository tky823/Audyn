import warnings
from typing import Any, Callable, Optional

import torch
from torch.utils.data import BatchSampler, DataLoader, Sampler

from .dataset import TrainingMammalDataset


class WordNetDataLoader(DataLoader):
    """DataLoader to support burnin parameter in TrainingMammalDataset.

    Args:
        burnin_step (int): Step to activate burnin parameter of dataset at ``burnin_step``th epoch.
            Default: ``0``.

    Examples:

        >>> from audyn.utils.data.wordnet import TrainingMammalDataset, WordNetDataLoader
        >>> dataset = TrainingMammalDataset(num_neg_samples=5, burnin_dampening=0.75)
        >>> dataloader = WordNetDataLoader(dataset, burnin_step=3)
        >>> dataloader.set_epoch(0)
        >>> dataset.burnin
        True
        >>> dataloader.set_epoch(1)
        >>> dataset.burnin
        True
        >>> dataloader.set_epoch(2)
        >>> dataset.burnin
        True
        >>> dataloader.set_epoch(3)
        >>> dataset.burnin
        False

    """

    dataset: TrainingMammalDataset

    def __init__(
        self,
        dataset: TrainingMammalDataset,
        batch_size: int = 1,
        shuffle: Optional[bool] = None,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        collate_fn: Callable = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable = None,
        multiprocessing_context: Any = None,
        generator: Optional[torch.Generator] = None,
        *,
        burnin_step: int = 0,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ) -> None:
        if not isinstance(dataset, TrainingMammalDataset):
            warnings.warn(
                "TrainingMammalDataset is expected as dataset, "
                f"but unexpected {type(dataset)} is given.",
                UserWarning,
                stacklevel=2,
            )

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

        self.burnin_step = burnin_step

    def set_epoch(self, epoch: int) -> None:
        burnin_step = self.burnin_step

        if burnin_step > epoch:
            self.dataset.set_burnin(True)
        else:
            self.dataset.set_burnin(False)
