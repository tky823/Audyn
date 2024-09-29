from typing import Optional

import torch.nn as nn

from ...models.music_tagging_transformer import MusicTaggingTransformer
from ...transforms.music_tagging_transformer import (
    MusicTaggingTransformerMelSpectrogram,
)

__all__ = [
    "music_tagging_transformer_melspectrogram",
    "music_tagging_transformer",
]


def music_tagging_transformer_melspectrogram() -> MusicTaggingTransformerMelSpectrogram:
    """Build MusicTaggingTransformerMelSpectrogram."""
    transform = MusicTaggingTransformerMelSpectrogram.build_from_pretrained()

    return transform


def music_tagging_transformer(
    role: str,
    aggregator: Optional[nn.Module] = None,
    head: Optional[nn.Module] = None,
) -> MusicTaggingTransformer:
    """Build MusicTaggingTransformer.

    Args:
        role (str): ``teacher`` or ``student``.
        aggregator (nn.Module, optional): Aggregator module.
        head (nn.Module, optional): Head module.

    """
    if role == "teacher":
        pretrained_model_name = "music-tagging-transformer_teacher"
    elif role == "student":
        pretrained_model_name = "music-tagging-transformer_student"
    else:
        raise ValueError(f"{role} is not supported as role.")

    model = MusicTaggingTransformer.build_from_pretrained(
        pretrained_model_name,
        aggregator=aggregator,
        head=head,
    )

    return model
