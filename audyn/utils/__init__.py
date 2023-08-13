import torch
from omegaconf import DictConfig

from .distributed import is_distributed, setup_distributed

__all__ = ["setup_system"]


def setup_system(config: DictConfig) -> None:
    """Set up system before training and evaluation.

    Args:
        config (DictConfig): Config to set up.

    """
    if is_distributed(config.system):
        setup_distributed(config.system)

    torch.manual_seed(config.system.seed)
