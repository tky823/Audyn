from typing import Any, Dict, List, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

__all__ = ["TransformerLRScheduler", "GANLRScheduler"]


class _DummyLRScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False) -> None:
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)


class TransformerLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        step_count = self._step_count

        d_model = self.d_model
        warmup_steps = self.warmup_steps

        lr = (d_model ** (-0.5)) * min(step_count ** (-0.5), step_count * warmup_steps ** (-1.5))
        return [lr for _ in self.optimizer.param_groups]


class MultiLRSchedulers:
    """Module to manage multiple learning rate schedulers."""

    # TODO: improve design

    def __init__(self, lr_schedulers: List[Union[Dict[str, Any], LRScheduler]]) -> None:
        self.lr_schedulers = {}

        for idx, lr_scheduler in enumerate(lr_schedulers):
            if isinstance(lr_scheduler, LRScheduler):
                k = str(idx)
                v = lr_scheduler
            elif isinstance(lr_scheduler, dict):
                k = lr_scheduler["name"]
                v = lr_scheduler["lr_scheduler"]
            else:
                raise ValueError(f"{type(lr_scheduler)} is not supported.")

            if k in self.lr_schedulers.keys():
                raise ValueError(f"Duplicate lr_scheduler name {k} is found.")

            self.lr_schedulers[k] = v

    def step(self, *args, **kwargs) -> None:
        for lr_scheduler in self.lr_schedulers.values():
            lr_scheduler: LRScheduler
            lr_scheduler.step(*args, **kwargs)

    def state_dict(self, *args, **kwargs) -> Dict[str, Dict[str, Any]]:
        state_dict = {}

        for name, lr_scheduler in self.lr_schedulers.items():
            lr_scheduler: LRScheduler
            state_dict[name] = lr_scheduler.state_dict(*args, **kwargs)

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        r"""Loads the learning rate scheduler state.

        Args:
            state_dict (dict): Learning rate scheduler state. Should be an object returned
                from a call to ``state_dict``.

        """

        for name, lr_scheduler in self.lr_schedulers.items():
            lr_scheduler: LRScheduler
            lr_scheduler.load_state_dict(state_dict[name])


class GANLRScheduler:
    def __init__(self, generator: LRScheduler, discriminator: LRScheduler) -> None:
        self.generator = generator
        self.discriminator = discriminator

    def step(self, *args, **kwargs) -> None:
        self.generator.step(*args, **kwargs)
        self.discriminator.step(*args, **kwargs)
