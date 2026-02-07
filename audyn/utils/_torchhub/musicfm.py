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


def musicfm_melspectrogram(
    dataset: int = "fma",
) -> MusicFMMelSpectrogram:
    """Build MusicTaggingTransformerMelSpectrogram.

    Args:
        dataset (str): Dataset in pretraining.

    Returns:
        MusicFMMelSpectrogram: MelSpectrogramTransform.

    """
    transform = MusicFMMelSpectrogram.build_from_pretrained(dataset=dataset)

    return transform


def musicfm(
    dataset: str = "fma",
    aggregator: Optional[nn.Module] = None,
    head: Optional[nn.Module] = None,
) -> MusicFM:
    """Build MusicFM.

    Args:
        dataset (str): Dataset in pretraining.
        aggregator (nn.Module, optional): Aggregator module.
        head (nn.Module, optional): Head module.

    Returns:
        MusicFM: Pretrained MusicFM.

    """
    if dataset == "fma":
        pretrained_model_name = "musicfm_fma"
    elif dataset == "msd":
        pretrained_model_name = "musicfm_msd"
    else:
        raise ValueError(f"{dataset} is not supported as dataset.")

    model = MusicFM.build_from_pretrained(
        pretrained_model_name,
        aggregator=aggregator,
        head=head,
    )

    return model
