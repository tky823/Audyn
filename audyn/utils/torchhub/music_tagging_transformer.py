from typing import Optional

import torch.nn as nn

from ...models.music_tagging_transformer import MusicTaggingTransformer

__all__ = [
    "teacher_music_tagging_transformer",
    "student_music_tagging_transformer",
]


def teacher_music_tagging_transformer(
    aggregator: Optional[nn.Module] = None,
    head: Optional[nn.Module] = None,
) -> MusicTaggingTransformer:
    """Build teacher MusicTaggingTransformer.

    Args:
        aggregator (nn.Module, optional): Aggregator module.
        head (nn.Module, optional): Head module.

    """
    is_teacher = True

    model = _music_tagging_transformer(
        is_teacher,
        aggregator=aggregator,
        head=head,
    )

    return model


def student_music_tagging_transformer(
    aggregator: Optional[nn.Module] = None,
    head: Optional[nn.Module] = None,
) -> MusicTaggingTransformer:
    """Build student MusicTaggingTransformer.

    Args:
        aggregator (nn.Module, optional): Aggregator module.
        head (nn.Module, optional): Head module.

    """
    is_teacher = False

    model = _music_tagging_transformer(
        is_teacher,
        aggregator=aggregator,
        head=head,
    )

    return model


def _music_tagging_transformer(
    is_teacher: bool,
    aggregator: Optional[nn.Module] = None,
    head: Optional[nn.Module] = None,
) -> MusicTaggingTransformer:
    """Build MusicTaggingTransformer.

    Args:
        is_teacher (bool): Teacher model or not.
        aggregator (nn.Module, optional): Aggregator module.
        head (nn.Module, optional): Head module.

    """
    if is_teacher:
        pretrained_model_name = "music-tagging-transformer_teacher"
    else:
        pretrained_model_name = "music-tagging-transformer_student"

    model = MusicTaggingTransformer.build_from_pretrained(
        pretrained_model_name,
        aggregator=aggregator,
        head=head,
    )

    return model
