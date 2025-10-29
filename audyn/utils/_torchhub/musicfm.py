from typing import Optional

import torch.nn as nn

from ...models.musicfm import MusicFM
from ...transforms.musicfm import (
    MusicFMMelSpectrogram,
)

__all__ = [
    "musicfm_melspectrogram",
    "musicfm",
]


def musicfm_melspectrogram() -> MusicFMMelSpectrogram:
    """Build MusicTaggingTransformerMelSpectrogram."""
    transform = MusicFMMelSpectrogram.build_from_pretrained()

    return transform


def musicfm(
    dataset: int = "msd",
    aggregator: Optional[nn.Module] = None,
    head: Optional[nn.Module] = None,
) -> MusicFM:
    """Build MusicFM.

    Args:
        dataset (int): Dataset in pretraining.
        aggregator (nn.Module, optional): Aggregator module.
        head (nn.Module, optional): Head module.

    Returns:
        MusicFM: Pretrained MusicFM.

    """
    if dataset == "msd":
        pretrained_model_name = "musicfm_msd"
    else:
        raise ValueError(f"{dataset} is not supported as dataset.")

    model = MusicFM.build_from_pretrained(
        pretrained_model_name,
        aggregator=aggregator,
        head=head,
    )

    return model
