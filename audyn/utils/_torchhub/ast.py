from typing import Optional

import torch.nn as nn
from torch.nn.common_types import _size_2_t

from ...models.ast import AudioSpectrogramTransformer

__all__ = [
    "ast_base",
]


def ast_base(
    pretrained_stride: int = 10,
    stride: Optional[_size_2_t] = None,
    n_bins: Optional[int] = None,
    n_frames: Optional[int] = None,
    aggregator: Optional[nn.Module] = None,
    head: Optional[nn.Module] = None,
) -> AudioSpectrogramTransformer:
    """Build AudioSpectrogramTransformer.

    Args:
        pretrained_stride (int): Stride used in pretraining.
        aggregator (nn.Module, optional): Aggregator module.
        head (nn.Module, optional): Head module.

    """
    if pretrained_stride == 10:
        pretrained_model_name = "ast-base-stride10"
    else:
        raise ValueError(f"{pretrained_stride} is not supported as pretrained_stride.")

    model = AudioSpectrogramTransformer.build_from_pretrained(
        pretrained_model_name,
        stride=stride,
        n_bins=n_bins,
        n_frames=n_frames,
        aggregator=aggregator,
        head=head,
    )

    return model
