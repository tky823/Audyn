from typing import Dict, Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..data import BaseDataLoaders
from .base import BaseTrainer


class TextToFeatTrainer(BaseTrainer):
    """Trainer for text-to-feat model."""

    def __init__(
        self,
        loaders: BaseDataLoaders,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: Optional[_LRScheduler] = None,
        criterion: Dict[str, nn.Module] = None,
        config: DictConfig = None,
    ) -> None:
        super().__init__(
            loaders,
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            config=config,
        )
