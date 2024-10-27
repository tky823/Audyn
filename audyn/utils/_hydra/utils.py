import itertools
from typing import Any, Dict, Iterable, Optional, Union, overload

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ...criterion.base import BaseCriterionWrapper, MultiCriteria
from ...metrics.base import StatefulMetric
from ...models.text_to_wave import CascadeTextToWave
from ...modules.rvq import ResidualVectorQuantizer
from ...modules.vqvae import VectorQuantizer
from ...optim.lr_scheduler import MultiLRSchedulers, _DummyLRScheduler
from ...optim.optimizer import ExponentialMovingAverageCodebookOptimizer, MultiOptimizers
from ..clip_grad import GradClipper
from ..parallel import is_dp_or_ddp

__all__ = [
    "instantiate",
    "instantiate_model",
    "instantiate_gan_generator",
    "instantiate_gan_discriminator",
    "instantiate_cascade_text_to_wave",
    "instantiate_optimizer",
    "instantiate_lr_scheduler",
    "instantiate_grad_clipper",
    "instantiate_criterion",
    "instantiate_metrics",
]

TORCH_CLIP_GRAD_FN = ["torch.nn.utils.clip_grad_value_", "torch.nn.utils.clip_grad_norm_"]

try:
    from torch.optim.optimizer import ParamsT

    optimizer_args_type = "ParamsT"

except ImportError:
    try:
        from torch.optim.optimizer import params_t

        optimizer_args_type = "params_t"
    except ImportError:
        optimizer_args_type = "Iterable"


if optimizer_args_type == "ParamsT":

    @overload
    def instantiate_optimizer(
        config: Union[DictConfig, ListConfig],
        module_or_params: Union[ParamsT, nn.Module],
        *args,
        **kwargs,
    ) -> Optimizer: ...

    @overload
    def instantiate_grad_clipper(
        config: DictConfig, module_or_params: ParamsT, *args, **kwargs
    ) -> GradClipper: ...

elif optimizer_args_type == "params_t":

    @overload
    def instantiate_optimizer(
        config: Union[DictConfig, ListConfig],
        module_or_params: Union[params_t, nn.Module],
        *args,
        **kwargs,
    ) -> Optimizer: ...

    @overload
    def instantiate_grad_clipper(
        config: DictConfig, module_or_params: params_t, *args, **kwargs
    ) -> GradClipper: ...

else:

    @overload
    def instantiate_optimizer(
        config: Union[DictConfig, ListConfig],
        module_or_params: Union[Iterable, nn.Module],
        *args,
        **kwargs,
    ) -> Optimizer: ...

    @overload
    def instantiate_grad_clipper(
        config: DictConfig, module_or_params: Iterable, *args, **kwargs
    ) -> GradClipper: ...


def instantiate(config: Any, *args, **kwargs) -> Any:
    """Wrapper function of ``hydra.utils.instantiate``."""
    return hydra.utils.instantiate(config, *args, **kwargs)


def instantiate_model(
    config_or_path: Union[str, DictConfig],
    *args,
    load_weights: Optional[bool] = None,
    **kwargs,
) -> nn.Module:
    """Instantiate model.

    Args:
        config_or_path (str or DictConfig): Config of model.
        args: Positional arguments given to ``instantiate``.
        load_weights (bool, optional): If ``True``, model loads pretrained weights.
            Default: ``False``.
        kwargs: Keyword arguments given to ``instantiate``.

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
        model: nn.Module = instantiate(model_config, *args, **kwargs)

        if load_weights:
            model.load_state_dict(state_dict["model"])

    elif isinstance(config_or_path, DictConfig):
        if load_weights:
            raise ValueError(
                "load_weights=True is not supported when config_or_path is DictConfig."
            )

        model_config = config_or_path
        model: nn.Module = instantiate(model_config, *args, **kwargs)
    else:
        raise NotImplementedError(f"{type(config_or_path)} is not supported.")

    return model


def instantiate_gan_generator(
    config_or_path: Union[str, DictConfig],
    load_weights: Optional[bool] = None,
) -> nn.Module:
    """Instantiate generator in GANs.

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
        generator_config = OmegaConf.create(model_config["generator"])
        generator: nn.Module = instantiate(generator_config)

        if load_weights:
            generator.load_state_dict(state_dict["model"]["generator"])

    elif isinstance(config_or_path, DictConfig):
        if load_weights:
            raise ValueError(
                "load_weights=True is not supported when config_or_path is DictConfig."
            )

        model_config = config_or_path

        if hasattr(model_config, "generator") and hasattr(model_config, "discriminator"):
            generator_config = model_config.generator
        else:
            generator_config = model_config

        generator: nn.Module = instantiate(generator_config)
    else:
        raise NotImplementedError(f"{type(config_or_path)} is not supported.")

    return generator


def instantiate_gan_discriminator(
    config_or_path: Union[str, DictConfig],
    load_weights: Optional[bool] = None,
) -> nn.Module:
    """Instantiate discriminator in GANs.

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
        discriminator_config = OmegaConf.create(model_config["discriminator"])
        discriminator: nn.Module = instantiate(discriminator_config)

        if load_weights:
            discriminator.load_state_dict(state_dict["model"]["discriminator"])

    elif isinstance(config_or_path, DictConfig):
        if load_weights:
            raise ValueError(
                "load_weights=True is not supported when config_or_path is DictConfig."
            )

        model_config = config_or_path

        if hasattr(model_config, "generator") and hasattr(model_config, "discriminator"):
            discriminator_config = model_config.discriminator
        else:
            discriminator_config = model_config

        discriminator: nn.Module = instantiate(discriminator_config)
    else:
        raise NotImplementedError(f"{type(config_or_path)} is not supported.")

    return discriminator


def instantiate_cascade_text_to_wave(
    config: DictConfig,
    text_to_feat_checkpoint: str,
    feat_to_wave_checkpoint: str,
    load_weights: Optional[bool] = None,
) -> "CascadeTextToWave":
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

    if hasattr(text_to_feat_model_config, "generator") and hasattr(
        text_to_feat_model_config, "discriminator"
    ):
        # We assume generator is used as text-to-feat model.
        is_text_to_feat_gan = True
        text_to_feat: nn.Module = instantiate(text_to_feat_model_config.generator)
    else:
        is_text_to_feat_gan = False
        text_to_feat: nn.Module = instantiate(text_to_feat_model_config)

    # feat-to-wave
    feat_to_wave_resolved_config: Dict[str, Any] = feat_to_wave_state_dict["resolved_config"]
    feat_to_wave_model_config: Dict[str, Any] = feat_to_wave_resolved_config["model"]
    feat_to_wave_model_config = OmegaConf.create(feat_to_wave_model_config)

    if hasattr(feat_to_wave_model_config, "generator") and hasattr(
        feat_to_wave_model_config, "discriminator"
    ):
        # We assume generator is used as feat-to-wave model.
        is_feat_to_wave_gan = True
        feat_to_wave: nn.Module = instantiate(feat_to_wave_model_config.generator)
    else:
        is_feat_to_wave_gan = False
        feat_to_wave: nn.Module = instantiate(feat_to_wave_model_config)

    model: CascadeTextToWave = instantiate(
        config,
        text_to_feat=text_to_feat,
        feat_to_wave=feat_to_wave,
    )

    if load_weights:
        if is_text_to_feat_gan:
            model.text_to_feat.load_state_dict(text_to_feat_state_dict["model"]["generator"])
        else:
            model.text_to_feat.load_state_dict(text_to_feat_state_dict["model"])

        if is_feat_to_wave_gan:
            model.feat_to_wave.load_state_dict(feat_to_wave_state_dict["model"]["generator"])
        else:
            model.feat_to_wave.load_state_dict(feat_to_wave_state_dict["model"])

    return model


def instantiate_optimizer(
    config, module_or_params, *args, **kwargs
) -> Union[Optimizer, MultiOptimizers]:
    """Instantiate optimizer."""

    def _register_forward_hook_for_ema_codebook_optim(module: nn.Module, optimizer: Optimizer):
        assert isinstance(module, (VectorQuantizer, ResidualVectorQuantizer)), (
            "Only VectorQuantizer and ResidualVectorQuantizer are supported "
            "for ExponentialMovingAverageCodebookOptimizer."
        )
        assert isinstance(optimizer, ExponentialMovingAverageCodebookOptimizer)

        module: Union[VectorQuantizer, ResidualVectorQuantizer]
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
                    if is_dp_or_ddp(module):
                        unwrapped_module = module.module
                    else:
                        unwrapped_module = module

                    submodule: nn.Module = getattr(unwrapped_module, submodule_name)
                    params.append(submodule.parameters())

                params = itertools.chain(*params)
                optimizer = instantiate(subconfig["optimizer"], params, *args, **kwargs)

                if isinstance(optimizer, ExponentialMovingAverageCodebookOptimizer):
                    for submodule_name in subconfig["modules"]:
                        if is_dp_or_ddp(module):
                            unwrapped_module = module.module
                        else:
                            unwrapped_module = module

                        submodule: nn.Module = getattr(unwrapped_module, submodule_name)
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
            optimizer = instantiate(config, params, *args, **kwargs)

            if isinstance(optimizer, ExponentialMovingAverageCodebookOptimizer):
                _register_forward_hook_for_ema_codebook_optim(module, optimizer)
    else:
        params = module_or_params

        if isinstance(config, ListConfig):
            raise ValueError("ListConfig is not supported when parameters are given to optimizer.")

        optimizer = instantiate(config, params, *args, **kwargs)

    return optimizer


def instantiate_lr_scheduler(
    config: DictConfig, optimizer: Union[Optimizer, MultiOptimizers], *args, **kwargs
) -> Optional[_LRScheduler]:
    """Instantiate learning rate scheduler.

    .. note::

        If ``config`` is empty dict, this function returns ``None``
        unlike ``instantiate``.

    """
    if isinstance(config, ListConfig):
        lr_schedulers = []

        if not isinstance(optimizer, MultiOptimizers):
            raise ValueError(
                f"MultiOptimizers are expected, but {type(optimizer)} is given as optimizer."
            )

        for idx, _config in enumerate(config):
            if "name" in _config:
                optim_name = _config["name"]
                _config = _config["lr_scheduler"]
            else:
                optim_name = idx

            _optimizer = optimizer.optimizers[optim_name]
            _lr_scheduler = instantiate(_config, _optimizer, *args, **kwargs)

            if _lr_scheduler is None or isinstance(_lr_scheduler, DictConfig):
                _lr_scheduler = _DummyLRScheduler(_optimizer)

            lr_schedulers.append({"name": optim_name, "lr_scheduler": _lr_scheduler})

        lr_schedulers = MultiLRSchedulers(lr_schedulers)

        return lr_schedulers
    else:
        lr_scheduler = instantiate(config, optimizer, *args, **kwargs)

        if lr_scheduler is None or isinstance(lr_scheduler, DictConfig):
            lr_scheduler = _DummyLRScheduler(optimizer)

    return lr_scheduler


def instantiate_grad_clipper(config, params, *args, **kwargs) -> GradClipper:
    if hasattr(config, "_target_"):
        if config._target_ in TORCH_CLIP_GRAD_FN:
            # for backward compatibility
            if config._target_ == "torch.nn.utils.clip_grad_value_":
                mode = "value"
            elif config._target_ == "torch.nn.utils.clip_grad_norm_":
                mode = "norm"

            overridden_config = OmegaConf.to_container(config)
            overridden_config.update({"_target_": "audyn.utils.GradClipper", "mode": mode})
            overridden_config = OmegaConf.create(overridden_config)
        else:
            overridden_config = config

        grad_clipper = instantiate(overridden_config, params, *args, **kwargs)

        if isinstance(grad_clipper, DictConfig):
            grad_clipper = None
    else:
        grad_clipper = None

    return grad_clipper


def instantiate_criterion(
    config: Union[DictConfig, ListConfig], *args, **kwargs
) -> Optional[nn.Module]:
    """Instantiate criterion."""

    if isinstance(config, DictConfig):
        criterion = instantiate(config, *args, **kwargs)
    elif isinstance(config, ListConfig):
        assert len(args) == 0, "Positional arguments are not supported."
        assert len(kwargs) == 0, "Keyword arguments are not supported."

        criteria_kwargs = {}

        for _config in config:
            _name = _config.name
            _criterion = instantiate(_config.criterion)
            _weight = _config.weight
            _key_mapping = OmegaConf.to_object(_config.key_mapping)
            _criterion_wrapper = BaseCriterionWrapper(
                _criterion, key_mapping=_key_mapping, weight=_weight
            )

            criteria_kwargs.update({_name: _criterion_wrapper})

        criterion = MultiCriteria(**criteria_kwargs)
    else:
        raise TypeError(f"Invalid type of config ({type(config)}) is specified.")

    return criterion


def instantiate_metrics(
    config: Union[DictConfig, ListConfig], *args, **kwargs
) -> Optional[StatefulMetric]:
    """Instantiate metrics.

    .. note::

        ``metrics`` is under beta version.

    """
    if isinstance(config, DictConfig):
        metrics = instantiate(config, *args, **kwargs)
    else:
        raise TypeError(f"Invalid type of config ({type(config)}) is specified.")

    return metrics
