from typing import Any

from torch.optim.optimizer import Optimizer


class DummyOptimizer(Optimizer):
    def __init__(self, params: Any) -> None:
        default = {"lr": 0}

        super().__init__(params, default)

    def step(self) -> None:
        pass
