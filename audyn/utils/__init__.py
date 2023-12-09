import importlib
import os
import warnings
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .clip_grad import GradClipper
from .data import select_accelerator
from .data.dataloader import (
    DistributedDataLoader,
    DistributedDynamicBatchDataLoader,
    DistributedSequentialBatchDataLoader,
    DynamicBatchDataLoader,
    SequentialBatchDataLoader,
)
from .distributed import is_distributed, setup_distributed
from .hydra.utils import (
    instantiate_cascade_text_to_wave,
    instantiate_criterion,
    instantiate_grad_clipper,
    instantiate_lr_scheduler,
    instantiate_model,
    instantiate_optimizer,
)

__all__ = [
    "setup_system",
    "instantiate_model",
    "instantiate_cascade_text_to_wave",
    "instantiate_optimizer",
    "instantiate_lr_scheduler",
    "instantiate_grad_clipper",
    "instantiate_criterion",
    "GradClipper",
]


def setup_system(config: DictConfig) -> None:
    """Set up system before training and evaluation.

    Args:
        config (DictConfig): Config to set up.

    """
    if hasattr(config, "system"):
        full_config = config
        system_config = config.system
    else:
        warnings.warn(
            "System config is given to setup_system. Full configuration is recommended.",
            DeprecationWarning,
            stacklevel=2,
        )
        full_config = None
        system_config = config

    accelerator = select_accelerator(system_config)

    if accelerator == "gpu":
        from torch.backends import cudnn

        cudnn.benchmark = system_config.cudnn.benchmark
        cudnn.deterministic = system_config.cudnn.deterministic

    if is_distributed(system_config):
        if full_config is None:
            warnings.warn(
                "System config is given to setup_system. In that case, "
                "training configuration is not converted to DDP.",
                UserWarning,
                stacklevel=2,
            )
        else:
            # overwrite full_config
            convert_dataloader_to_ddp_if_possible(full_config)
            system_config = full_config.system

        setup_distributed(system_config)

    torch.manual_seed(system_config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)


def convert_dataloader_to_ddp_if_possible(config: DictConfig) -> None:
    """Convert data loader in config.train.dataloader.train for DDP.

    .. note::

        This function may overwrite config.train.dataloader.train.

    .. note::

        If conversion is required, you have to set environmental variables
        ``WORLD_SIZE`` and ``RANK``.

    """
    train_dataloader_config = config.train.dataloader.train

    # split _target_ into names of package, module, variable
    # e.g.
    #     _target_: audyn.utils.data.SequentialBatchDataLoader
    # package_name: audyn
    #     mod_name: audyn.utils.data
    #     var_name: SequentialBatchDataLoader
    mod_name, var_name = train_dataloader_config._target_.rsplit(".", maxsplit=1)
    package_name, *_ = mod_name.split(".", maxsplit=1)
    cls = getattr(importlib.import_module(mod_name), var_name)

    if package_name == "torch":
        if cls is DataLoader:
            # may be converted to distributed data loader
            if "sampler" in train_dataloader_config.keys():
                # TODO: torch.utils.data.DistributedSampler is expected to work well.
                sampler = train_dataloader_config.sampler

                if sampler is not None:
                    raise ValueError("Sampler cannot be automatically converted to DDP-supported.")

            train_dataloader_config = OmegaConf.to_container(train_dataloader_config)

            if "seed" in train_dataloader_config.keys():
                # NOTE: Since torch.utils.data.DataLoader does not support seed,
                #       seed should not be defined in config.train.dataloader.train for proper use.
                seed = train_dataloader_config["seed"]
            else:
                seed = "${system.seed}"

            # DataLoader -> DistributedDataLoader
            ddp_target = ".".join(
                [DistributedDataLoader.__module__, DistributedDataLoader.__name__]
            )
            additional_ddp_config = {
                "_target_": ddp_target,
                "num_replicas": int(os.environ["WORLD_SIZE"]),
                "rank": int(os.environ["RANK"]),
                "seed": seed,
            }
            train_dataloader_config.update(additional_ddp_config)
            OmegaConf.update(
                config, "train.dataloader.train", train_dataloader_config, merge=False
            )
        else:
            _warn_unexpected_dataloader_for_ddp(cls)
    elif package_name == "audyn":
        if cls is SequentialBatchDataLoader or cls is DynamicBatchDataLoader:
            train_dataloader_config = OmegaConf.to_container(train_dataloader_config)

            if "seed" in train_dataloader_config.keys():
                seed = train_dataloader_config["seed"]
            else:
                seed = "${system.seed}"

            # should be converted to distributed data loader
            # SequentialBatchDataLoader -> DistributedSequentialBatchDataLoader
            # DynamicBatchDataLoader -> DistributedDynamicBatchDataLoader
            ddp_target = ".".join([mod_name, "Distributed" + cls.__name__])
            additional_ddp_config = {
                "_target_": ddp_target,
                "num_replicas": int(os.environ["WORLD_SIZE"]),
                "rank": int(os.environ["RANK"]),
                "seed": seed,
            }
            train_dataloader_config.update(additional_ddp_config)
            OmegaConf.update(
                config, "train.dataloader.train", train_dataloader_config, merge=False
            )
        elif (
            cls is DistributedDataLoader
            or cls is DistributedSequentialBatchDataLoader
            or cls is DistributedDynamicBatchDataLoader
        ):
            # These data loaders support DDP.
            pass
        else:
            _warn_unexpected_dataloader_for_ddp(cls)
    else:
        _warn_unexpected_dataloader_for_ddp(cls)


def _warn_unexpected_dataloader_for_ddp(module: Any) -> None:
    """Warn unexpected data loader that cannot be converted to DDP-supported one.

    Both torch's and audyn's data loaders are supported.
    """
    warnings.warn(
        f"Unexpected data loader {module.__name__} is found. "
        "There is not rule to convert to DDP-supported loader.",
        UserWarning,
        stacklevel=2,
    )
