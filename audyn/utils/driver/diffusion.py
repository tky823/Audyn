from typing import Dict, Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from audyn.utils.clip_grad import GradClipper
from audyn.utils.data import BaseDataLoaders

from ..diffusion.sampler import ReverseSampler
from ..hydra.utils import instantiate
from .base import BaseTrainer


class DenoisingDiffusionTrainer(BaseTrainer):

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
            loaders=loaders,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_clipper=grad_clipper,
            criterion=criterion,
            config=config,
        )

        if not hasattr(config.train, "diffusion"):
            raise ValueError("config.train.diffusion is required.")

        if not hasattr(config.train.diffusion, "reverse_sampler"):
            raise ValueError("config.train.diffusion.reverse_sampler is required.")

        self.reverse_sampler: ReverseSampler = instantiate(
            config.train.diffusion.reverse_sampler, model
        )

    def train_one_epoch(self) -> Dict[str, float]:
        return super().train_one_epoch()
