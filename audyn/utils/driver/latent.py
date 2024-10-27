from typing import Dict, Optional, Union

import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ...models.vae import BaseVAE
from ..data import BaseDataLoaders
from .base import BaseTrainer


class LatentTrainer(BaseTrainer):
    def __init__(
        self,
        loaders: BaseDataLoaders,
        model: nn.Module,
        vae: Union[BaseVAE, nn.Module],
        optimizer: Optimizer,
        lr_scheduler: Optional[_LRScheduler] = None,
        criterion: Dict[str, nn.Module] = None,
        config: DictConfig = None,
    ) -> None:
        super().__init__(
            loaders,
            model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            config=config,
        )

        self.vae = vae

    def train_one_epoch(self) -> Dict[str, float]:
        sampled = self.vae.rsample(**named_input)

        output = self.unwrapped_model(**named_input)

        return super().train_one_epoch()
