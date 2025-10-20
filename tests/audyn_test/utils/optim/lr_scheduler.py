from typing import Any, List

import torch
from packaging import version
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

IS_TORCH_GTE_2_7 = version.parse(torch.__version__) >= version.parse("2.7")


class DummyLRScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
        if IS_TORCH_GTE_2_7:
            verbose = kwargs.pop("verbose", False)

            assert not verbose, "verbose=True is not supported."

        super().__init__(
            optimizer,
            last_epoch=last_epoch,
            **kwargs,
        )

    def get_lr(self) -> List[Any]:
        return [param_group["lr"] for param_group in self.optimizer.param_groups]
