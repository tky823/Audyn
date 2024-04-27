from typing import Optional

import torch.nn as nn
from torch.nn.common_types import _size_2_t

from audyn.models.ast import AudioSpectrogramTransformer as _AST
from audyn.models.passt import PaSST as _PaSST
from audyn.models.ssast import (
    MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel as _MultiTaskSSASTMPM,
)
from audyn.models.ssast import SelfSupervisedAudioSpectrogramTransformer as _SSAST


def ast_base(
    pretrained_stride: int = 10,
    stride: Optional[_size_2_t] = None,
    n_bins: Optional[int] = None,
    n_frames: Optional[int] = None,
    aggregator: Optional[nn.Module] = None,
    head: Optional[nn.Module] = None,
) -> _AST:
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

    model = _AST.build_from_pretrained(
        pretrained_model_name,
        stride=stride,
        n_bins=n_bins,
        n_frames=n_frames,
        aggregator=aggregator,
        head=head,
    )

    return model


def multitask_ssast_base_400(
    token_unit: str = "patch",
    stride: Optional[_size_2_t] = None,
    n_bins: Optional[int] = None,
    n_frames: Optional[int] = None,
    reconstructor: Optional[nn.Module] = None,
    classifier: Optional[nn.Module] = None,
) -> _MultiTaskSSASTMPM:
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

    model = _MultiTaskSSASTMPM.build_from_pretrained(
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
) -> _SSAST:
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

    model = _SSAST.build_from_pretrained(
        pretrained_model_name,
        stride=stride,
        n_bins=n_bins,
        n_frames=n_frames,
        aggregator=aggregator,
        head=head,
    )

    return model


def passt_base_ap476_swa(
    stride: int = 10,
    patchout: str = "struct",
    n_bins: Optional[int] = None,
    n_frames: Optional[int] = None,
    aggregator: Optional[nn.Module] = None,
    head: Optional[nn.Module] = None,
) -> _PaSST:
    """Build SelfSupervisedAudioSpectrogramTransformer.

    Args:
        stride (int): Stride of patch.
        patchout (str): Type of patchout technique.
        aggregator (nn.Module, optional): Aggregator module.
        head (nn.Module, optional): Head module.

    """
    if stride == 10 and patchout == "struct":
        pretrained_model_name = f"passt-base-stride{stride}-{patchout}-ap0.476-swa"
    else:
        raise ValueError(f"Model satisfying stride={stride} and patchout={patchout} is not found.")

    model = _PaSST.build_from_pretrained(
        pretrained_model_name,
        stride=stride,
        n_bins=n_bins,
        n_frames=n_frames,
        aggregator=aggregator,
        head=head,
    )

    return model
