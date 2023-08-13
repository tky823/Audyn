import torch
from omegaconf import DictConfig

from .distributed import setup_distributed

__all__ = ["setup_system"]


def setup_system(config: DictConfig) -> None:
    """Set up system before training and evaluation.

    Args:
        config (DictConfig): Config to set up.

    """
    if config.system.distributed.enable:
        setup_distributed(config.system)

    torch.manual_seed(config.system.seed)
