import warnings

import torch
from omegaconf import DictConfig

from .clip_grad import GradClipper
from .data import select_accelerator
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
        system_config = config.system
    else:
        warnings.warn(
            "System config is given to setup_system. Full configuration is recommended.",
            DeprecationWarning,
            stacklevel=2,
        )
        system_config = config

    accelerator = select_accelerator(system_config)

    if accelerator == "gpu":
        from torch.backends import cudnn

        cudnn.benchmark = system_config.cudnn.benchmark
        cudnn.deterministic = system_config.cudnn.deterministic

    if is_distributed(system_config):
        warnings.warn(
            "System config is given to setup_system. In that case, "
            "training configuration is not converted to DDP.",
            UserWarning,
            stacklevel=2,
        )
        setup_distributed(system_config)

    torch.manual_seed(system_config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
