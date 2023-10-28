from typing import Any, Dict, Iterable, Optional, Union, overload

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from packaging import version
from torch.optim import Optimizer

from ...models.text_to_wave import CascadeTextToWave

__all__ = ["instantiate_model", "instantiate_cascade_text_to_wave"]

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


@overload
def instantiate_optimizer(config: DictConfig, params: Iterable, *args, **kwargs) -> Optimizer:
    ...


if not IS_TORCH_LT_2_1:
    from torch.optim.optimizer import params_t

    @overload
    def instantiate_optimizer(config: DictConfig, params: params_t, *args, **kwargs) -> Optimizer:
        ...


def instantiate_model(
    config_or_path: Union[str, DictConfig],
    load_weights: Optional[bool] = None,
) -> nn.Module:
    """Instantiate model.

    Args:
        config_or_path (str or DictConfig): Config of model.
        load_weights (bool, optional): If ``True``, model loads pretrained weights.
            Default: ``False``.

    Returns:
        nn.Module: Constructed model.

    """
    if load_weights is None:
        load_weights = False

    if isinstance(config_or_path, str):
        state_dict = torch.load(config_or_path, map_location=lambda storage, loc: storage)

        resolved_config: Dict[str, Any] = state_dict["resolved_config"]
        model_config: Dict[str, Any] = resolved_config["model"]
        model_config = OmegaConf.create(model_config)
        model: nn.Module = hydra.utils.instantiate(model_config)

        if load_weights:
            model.load_state_dict(state_dict["model"])

    elif isinstance(config_or_path, DictConfig):
        if load_weights:
            raise ValueError(
                "load_weights=True is not supported when config_or_path is DictConfig."
            )

        model_config = config_or_path
        model: nn.Module = hydra.utils.instantiate(model_config)
    else:
        raise NotImplementedError(f"{type(config_or_path)} is not supported.")

    return model


def instantiate_cascade_text_to_wave(
    config: DictConfig,
    text_to_feat_checkpoint: str,
    feat_to_wave_checkpoint: str,
    load_weights: Optional[bool] = None,
) -> CascadeTextToWave:
    """Instantiate cascade text-to-wave model.

    Args:
        config (DictConfig): Config of cascase text-to-wave model.
        text_to_feat_checkpoint (str): Path to pretrained text-to-feat model.
        feat_to_wave_checkpoint (str): Path to pretrained feat-to-wave model.
        load_weights (bool, optional): If ``True``, model loads pretrained weights.

    Returns:
        CascadeTextToWave: Constructed cascase text-to-wave model.

    """
    if load_weights is None:
        load_weights = False

    text_to_feat_state_dict = torch.load(
        text_to_feat_checkpoint, map_location=lambda storage, loc: storage
    )
    feat_to_wave_state_dict = torch.load(
        feat_to_wave_checkpoint, map_location=lambda storage, loc: storage
    )

    # text-to-feat
    text_to_feat_resolved_config: Dict[str, Any] = text_to_feat_state_dict["resolved_config"]
    text_to_feat_model_config: Dict[str, Any] = text_to_feat_resolved_config["model"]
    text_to_feat_model_config = OmegaConf.create(text_to_feat_model_config)
    text_to_feat: nn.Module = hydra.utils.instantiate(text_to_feat_model_config)

    # feat-to-wave
    feat_to_wave_resolved_config: Dict[str, Any] = feat_to_wave_state_dict["resolved_config"]
    feat_to_wave_model_config: Dict[str, Any] = feat_to_wave_resolved_config["model"]
    feat_to_wave_model_config = OmegaConf.create(feat_to_wave_model_config)
    feat_to_wave: nn.Module = hydra.utils.instantiate(feat_to_wave_model_config)

    model: CascadeTextToWave = hydra.utils.instantiate(
        config,
        text_to_feat=text_to_feat,
        feat_to_wave=feat_to_wave,
    )

    if load_weights:
        model.text_to_feat.load_state_dict(text_to_feat_state_dict["model"])
        model.feat_to_wave.load_state_dict(feat_to_wave_state_dict["model"])

    return model


def instantiate_optimizer(config, params, *args, **kwargs) -> Optimizer:
    return hydra.utils.instantiate(config, params, *args, **kwargs)
