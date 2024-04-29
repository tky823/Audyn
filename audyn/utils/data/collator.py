from typing import Any, List, Optional

from . import default_collate_fn
from .dataset import Composer

__all__ = [
    "Collator",
]


class Collator:
    """Base class of collator."""

    def __init__(self, composer: Optional[Composer] = None) -> None:
        self.composer = composer

    def __call__(self, batch: List[Any]) -> None:
        composer = self.composer

        if composer is None:
            list_batch = batch
        else:
            list_batch = []

            for sample in composer(batch):
                list_batch.append(sample)

        dict_batch = default_collate_fn(list_batch)

        return dict_batch
