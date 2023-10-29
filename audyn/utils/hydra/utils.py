import itertools
from typing import Any, Dict, Iterable, Optional, Union, overload

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf
from packaging import version
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ...models.text_to_wave import CascadeTextToWave
from ...modules.vqvae import VectorQuantizer
from ...optim.optimizer import ExponentialMovingAverageCodebookOptimizer, MultiOptimizers

__all__ = [
    "instantiate_model",
    "instantiate_cascade_text_to_wave",
    "instantiate_optimizer",
    "instantiate_lr_scheduler",
]

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


if IS_TORCH_LT_2_1:

    @overload
    def instantiate_optimizer(
        config: Union[DictConfig, ListConfig],
        module_or_params: Union[Iterable, nn.Module],
        *args,
        **kwargs,
    ) -> Optimizer:
        ...

else:
    from torch.optim.optimizer import params_t

    @overload
    def instantiate_optimizer(
        config: Union[DictConfig, ListConfig],
        module_or_params: Union[params_t, nn.Module],
        *args,
        **kwargs,
    ) -> Optimizer:
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


def instantiate_optimizer(
    config, module_or_params, *args, **kwargs
) -> Union[Optimizer, MultiOptimizers]:
    """Instantiate optimizer."""

    def _register_forward_hook_for_ema_codebook_optim(module: nn.Module, optimizer: Optimizer):
        assert isinstance(module, VectorQuantizer), (
            "Only VectorQuantizer is supported " "for ExponentialMovingAverageCodebookOptimizer."
        )
        assert isinstance(optimizer, ExponentialMovingAverageCodebookOptimizer)

        module: VectorQuantizer
        optimizer: ExponentialMovingAverageCodebookOptimizer
        module.register_forward_hook(optimizer.store_current_stats)

    if isinstance(module_or_params, nn.Module):
        module = module_or_params

        if isinstance(config, ListConfig):
            optimizers = []

            for idx, subconfig in enumerate(config):
                name = subconfig.get("name", f"{idx}")
                params = []

                for submodule_name in subconfig["modules"]:
                    submodule: nn.Module = getattr(module, submodule_name)
                    params.append(submodule.parameters())

                params = itertools.chain(*params)
                optimizer = hydra.utils.instantiate(
                    subconfig["optimizer"], params, *args, **kwargs
                )

                if isinstance(optimizer, ExponentialMovingAverageCodebookOptimizer):
                    for submodule_name in subconfig["modules"]:
                        submodule: nn.Module = getattr(module, submodule_name)
                        _register_forward_hook_for_ema_codebook_optim(submodule, optimizer)

                optimizer = {
                    "name": name,
                    "optimizer": optimizer,
                }
                optimizers.append(optimizer)

            optimizers = MultiOptimizers(optimizers)

            return optimizers
        else:
            params = module.parameters()
            optimizer = hydra.utils.instantiate(config, params, *args, **kwargs)

            if isinstance(optimizer, ExponentialMovingAverageCodebookOptimizer):
                _register_forward_hook_for_ema_codebook_optim(module, optimizer)
    else:
        params = module_or_params

        if isinstance(config, ListConfig):
            raise ValueError("ListConfig is not supported when parameters are given to optimizer.")

        optimizer = hydra.utils.instantiate(config, params, *args, **kwargs)

    return optimizer


def instantiate_lr_scheduler(
    config: DictConfig, optimizer: Optimizer, *args, **kwargs
) -> Optional[_LRScheduler]:
    """Instantiate learning rate scheduler.

    .. note::

        If ``config`` is empty dict, this function returns ``None``
        unlike ``hydra.utils.instantiate``.

    """
    lr_scheduler = hydra.utils.instantiate(config, optimizer, *args, **kwargs)

    if isinstance(lr_scheduler, DictConfig):
        lr_scheduler = None

    return lr_scheduler
