from typing import Any, List

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class DummyLRScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False) -> None:
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self) -> List[Any]:
        return [param_group["lr"] for param_group in self.optimizer.param_groups]
