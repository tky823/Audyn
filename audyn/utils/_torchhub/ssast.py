from typing import Optional

import torch.nn as nn
from torch.nn.common_types import _size_2_t

from ...models.ssast import (
    MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel as MultiTaskSSASTMPM,
)
from ...models.ssast import SelfSupervisedAudioSpectrogramTransformer as SSAST


def multitask_ssast_base_400(
    token_unit: str = "patch",
    stride: Optional[_size_2_t] = None,
    n_bins: Optional[int] = None,
    n_frames: Optional[int] = None,
    reconstructor: Optional[nn.Module] = None,
    classifier: Optional[nn.Module] = None,
) -> MultiTaskSSASTMPM:
    """Build SelfSupervisedAudioSpectrogramTransformer.

    Args:
        token_unit (str): Token unit. ``patch`` and ``frame`` are supported.

    """
    if token_unit == "patch":
        pretrained_model_name = "multitask-ssast-patch-base-400"
    elif token_unit == "frame":
        pretrained_model_name = "multitask-ssast-frame-base-400"
    else:
        raise ValueError(f"{token_unit} is not supported as token_unit.")

    model = MultiTaskSSASTMPM.build_from_pretrained(
        pretrained_model_name,
        stride=stride,
        n_bins=n_bins,
        n_frames=n_frames,
        reconstructor=reconstructor,
        classifier=classifier,
    )

    return model


def ssast_base_400(
    token_unit: str = "patch",
    stride: Optional[_size_2_t] = None,
    n_bins: Optional[int] = None,
    n_frames: Optional[int] = None,
    aggregator: Optional[nn.Module] = None,
    head: Optional[nn.Module] = None,
) -> SSAST:
    """Build SelfSupervisedAudioSpectrogramTransformer.

    Args:
        token_unit (str): Token unit. ``patch`` and ``frame`` are supported.
        aggregator (nn.Module, optional): Aggregator module.
        head (nn.Module, optional): Head module.

    """
    if token_unit == "patch":
        pretrained_model_name = "multitask-ssast-patch-base-400"
    elif token_unit == "frame":
        pretrained_model_name = "multitask-ssast-frame-base-400"
    else:
        raise ValueError(f"{token_unit} is not supported as token_unit.")

    model = SSAST.build_from_pretrained(
        pretrained_model_name,
        stride=stride,
        n_bins=n_bins,
        n_frames=n_frames,
        aggregator=aggregator,
        head=head,
    )

    return model
