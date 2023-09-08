import torch
from omegaconf import DictConfig

from .data import select_accelerator
from .distributed import is_distributed, setup_distributed
from .hydra.utils import instantiate_cascade_text_to_wave, instantiate_model

__all__ = [
    "setup_system",
    "instantiate_model",
    "instantiate_cascade_text_to_wave",
]


def setup_system(config: DictConfig) -> None:
    """Set up system before training and evaluation.

    Args:
        config (DictConfig): Config to set up.

    """
    if hasattr(config, "system"):
        system_config = config.system
    else:
        system_config = config

    accelerator = select_accelerator(system_config)

    if accelerator == "gpu":
        from torch.backends import cudnn

        cudnn.benchmark = True
        cudnn.deterministic = False

    if is_distributed(system_config):
        setup_distributed(system_config)

    torch.manual_seed(system_config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
