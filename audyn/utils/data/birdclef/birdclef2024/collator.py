from typing import Optional

from ...composer import Composer
from .._common.collater import BirdCLEFCollator
from . import num_primary_labels as num_birdclef2024_primary_labels

__all__ = [
    "BirdCLEF2024BaselineCollator",
]


class BirdCLEF2024BaselineCollator(BirdCLEFCollator):
    def __init__(
        self,
        composer: Optional[Composer] = None,
        melspectrogram_key: str = "melspectrogram",
        label_index_key: str = "label_index",
        alpha: float = 0.4,
    ) -> None:
        super().__init__(
            composer=composer,
            melspectrogram_key=melspectrogram_key,
            label_index_key=label_index_key,
            num_primary_labels=num_birdclef2024_primary_labels,
            alpha=alpha,
        )
