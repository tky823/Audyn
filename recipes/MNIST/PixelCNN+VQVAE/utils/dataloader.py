from typing import Any

import torch
from torch.utils.data import DataLoader

from .sampler import GumbelVQVAERandomSampler


class GumbelVQVAEDataLoader(DataLoader):

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle: bool | None = None,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> None:
        sampler = GumbelVQVAERandomSampler(
            dataset,
            generator=generator,
        )
        batch_sampler = None

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
            **kwargs,
        )

    def __iter__(self, *args, **kwargs) -> Any:
        sampler: GumbelVQVAERandomSampler = self.sampler
        step = sampler.get_step()

        if step < 0:
            sampler.set_step(0)

        for sample in super().__iter__(*args, **kwargs):
            step = sampler.get_step()
            sampler.set_step(step + 1)

            yield sample
