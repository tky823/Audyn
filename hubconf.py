from typing import Optional

import torch.nn as nn
from torch.nn.common_types import _size_2_t

from audyn.models.ssast import (
    MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel as _MultiTaskSSASTMPM,
)
from audyn.models.ssast import SelfSupervisedAudioSpectrogramTransformer as _SSAST


def multitask_ssast_mpm(
    pretrained_model_name: str,
    stride: Optional[_size_2_t] = None,
    n_bins: Optional[int] = None,
    n_frames: Optional[int] = None,
    reconstructor: Optional[nn.Module] = None,
    classifier: Optional[nn.Module] = None,
) -> _MultiTaskSSASTMPM:
    """Build SelfSupervisedAudioSpectrogramTransformer.

    Args:
        pretrained_model_name (str): Pretrained model name.

    .. note::

        Supported pretrained model names are
            - multitask-ssast-patch-base-400
            - multitask-ssast-frame-base-400

    """
    model = _MultiTaskSSASTMPM.build_from_pretrained(
        pretrained_model_name,
        stride=stride,
        n_bins=n_bins,
        n_frames=n_frames,
        reconstructor=reconstructor,
        classifier=classifier,
    )

    return model


def ssast(
    pretrained_model_name: str,
    stride: Optional[_size_2_t] = None,
    n_bins: Optional[int] = None,
    n_frames: Optional[int] = None,
    aggregator: Optional[nn.Module] = None,
    head: Optional[nn.Module] = None,
) -> _SSAST:
    """Build SelfSupervisedAudioSpectrogramTransformer.

    Args:
        pretrained_model_name (str): Pretrained model name.
        aggregator (nn.Module, optional): Aggregator module.
        head (nn.Module, optional): Head module.

    .. note::

        Supported pretrained model names are
            - multitask-ssast-patch-base-400
            - multitask-ssast-frame-base-400

    """
    model = _SSAST.build_from_pretrained(
        pretrained_model_name,
        stride=stride,
        n_bins=n_bins,
        n_frames=n_frames,
        aggregator=aggregator,
        head=head,
    )

    return model
