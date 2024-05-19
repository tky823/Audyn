from typing import Any, Dict, List, Optional

import torchvision

from ...collator import Collator
from ...composer import Composer
from . import num_primary_labels as num_birdclef2024_primary_labels

__all__ = [
    "BirdCLEF2024BaselineCollator",
]


class BirdCLEF2024BaselineCollator(Collator):
    def __init__(
        self,
        composer: Optional[Composer] = None,
        melspectrogram_key: str = "melspectrogram",
        label_index_key: str = "label_index",
        alpha: float = 0.4,
    ) -> None:
        super().__init__(composer=composer)

        self.melspectrogram_key = melspectrogram_key
        self.label_index_key = label_index_key

        try:
            from torchvision.transforms.v2 import MixUp
        except ImportError:
            raise ImportError(f"MixUp is not supported by torchvision=={torchvision.__version__}")

        self.mixup = MixUp(
            alpha=alpha,
            num_classes=num_birdclef2024_primary_labels,
        )

    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        melspectrogram_key = self.melspectrogram_key
        label_index_key = self.label_index_key

        dict_batch = super().__call__(batch)
        melspectrogram = dict_batch[melspectrogram_key]
        label_index = dict_batch[label_index_key]

        # 4D input is required to mixup.
        *batch_shape, n_bins, n_frames = melspectrogram.size()
        melspectrogram = melspectrogram.view(-1, 1, n_bins, n_frames)
        melspectrogram, label_index = self.mixup(melspectrogram, label_index)
        melspectrogram = melspectrogram.view(*batch_shape, n_bins, n_frames)

        dict_batch[melspectrogram_key] = melspectrogram
        dict_batch[label_index_key] = label_index

        return dict_batch
