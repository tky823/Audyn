import os
from typing import Dict, Optional

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader

from ..clip_grad import GradClipper
from ..data import BaseDataLoaders
from ..logging import get_logger
from .base import BaseGenerator, BaseTrainer


class FeatToWaveTrainer(BaseTrainer):
    """Trainer for feat-to-wave model (a.k.a vocoder)."""

    def __init__(
        self,
        loaders: BaseDataLoaders,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: Optional[_LRScheduler] = None,
        grad_clipper: Optional[GradClipper] = None,
        criterion: Dict[str, nn.Module] = None,
        config: DictConfig = None,
    ) -> None:
        super().__init__(
            loaders,
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            grad_clipper=grad_clipper,
            criterion=criterion,
            config=config,
        )


class FeatToWaveGenerator(BaseGenerator):
    """Generator for feat-to-wave model (a.k.a vocoder)."""

    def __init__(
        self,
        loader: DataLoader,
        model: nn.Module,
        config: DictConfig = None,
    ) -> None:
        super().__init__(loader, model, config=config)

    def _reset(self, config: DictConfig) -> None:
        self.set_system(config=config.system)

        self.exp_dir = config.test.output.exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)

        self.inference_dir = config.test.output.inference_dir
        os.makedirs(self.inference_dir, exist_ok=True)

        # Set loggder
        self.logger = get_logger(
            self.__class__.__name__,
            is_distributed=self.is_distributed,
        )

        # Display config and model architecture after logger instantiation
        self.logger.info(OmegaConf.to_yaml(self.config))
        self.display_model(display_num_parameters=True)

        if hasattr(config.test.checkpoint, "feat_to_wave"):
            checkpoint = config.test.checkpoint.feat_to_wave
        else:
            checkpoint = config.test.checkpoint

        self.logger.info(f"Load weights of feat to wave model: {checkpoint}.")
        self.load_checkpoint(checkpoint)

        self.remove_weight_norm_if_necessary()
