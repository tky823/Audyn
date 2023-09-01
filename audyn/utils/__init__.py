import torch
from omegaconf import DictConfig

from .distributed import is_distributed, setup_distributed

__all__ = ["setup_system"]


def setup_system(config: DictConfig) -> None:
    """Set up system before training and evaluation.

    Args:
        config (DictConfig): Config to set up.

    """
    if hasattr(config, "system"):
        system_config = config.system
    else:
        system_config = config

    if is_distributed(system_config):
        setup_distributed(system_config)

    torch.manual_seed(system_config.seed)
