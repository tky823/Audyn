import math
from typing import Any, Dict, List, Optional, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

try:
    from torch.optim.lr_scheduler import LRScheduler as _LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler

__all__ = [
    "_DummyLRScheduler",
    "_DummyLR",
    "TransformerLRScheduler",
    "NoamScheduler",
    "TransformerLR",
    "NoamLR",
    "ExponentialWarmupLinearCooldownLRScheduler",
    "ExponentialWarmupLinearCooldownLR",
    "MultiLRSchedulers",
    "MultiLR",
    "GANLRScheduler",
    "GANLR",
]


class _DummyLRScheduler:
    """Dummy learning rate scheduler which does not change learning rate."""

    def __init__(self, optimizer: Optimizer, *args, **kwargs) -> None:
        self.optimizer = optimizer

    def step(self, *args, **kwargs) -> None:
        pass

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        assert len(state_dict) == 0


class _DummyLR(_DummyLRScheduler):
    """Alias of _DummyLRScheduler."""


class TransformerLRScheduler(_LRScheduler):
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


class NoamScheduler(TransformerLRScheduler):
    """Alias of TransformerLRScheduler."""


class TransformerLR(TransformerLRScheduler):
    """Alias of TransformerLRScheduler."""


class NoamLR(NoamScheduler):
    """Alias of NoamLR."""


class ExponentialWarmupLinearCooldownLRScheduler(LambdaLR):
    """Exponential warm-up + linear cool-down of learning rate.

    This learning rate schduler is used to train PaSST.

    Args:
        optimizer (Optimizer): Optimizer to adjust learning rate.
        warmup_steps (int): Number of exponential warm-up steps.
        suspend_steps (int): Number of constant learning rate steps between warm-up and cool-down.
        cooldown_steps (int): Number of linear cool-down steps after constant learning rate.
        last_factor (float): Scale factor of learning rate at last step.

    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        suspend_steps: int,
        cooldown_steps: int,
        last_factor: float = 1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        def _lr_scheduler_lambda(step: int) -> float:
            if step < warmup_steps:
                normalized_step = 1 - step / warmup_steps
                factor = math.exp(-5.0 * normalized_step**2)
            elif step < warmup_steps + suspend_steps:
                factor = 1
            elif step < warmup_steps + suspend_steps + cooldown_steps:
                step_after_suspend = step - (warmup_steps + suspend_steps)
                normalized_step = step_after_suspend / cooldown_steps
                factor = last_factor + (1 - last_factor) * normalized_step
            else:
                factor = last_factor

            return factor

        super().__init__(
            optimizer,
            lr_lambda=_lr_scheduler_lambda,
            last_epoch=last_epoch,
            verbose=verbose,
        )


class ExponentialWarmupLinearCooldownLR(ExponentialWarmupLinearCooldownLRScheduler):
    """Alias of ExponentialWarmupLinearCooldownLRScheduler."""


class MultiLRSchedulers:
    """Module to manage multiple learning rate schedulers."""

    # TODO: improve design

    def __init__(self, lr_schedulers: List[Union[Dict[str, Any], _LRScheduler]]) -> None:
        self.lr_schedulers = {}

        for idx, lr_scheduler in enumerate(lr_schedulers):
            if isinstance(lr_scheduler, _LRScheduler):
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
            lr_scheduler: Optional[_LRScheduler]

            if lr_scheduler is not None:
                lr_scheduler.step(*args, **kwargs)

    def state_dict(self, *args, **kwargs) -> Dict[str, Dict[str, Any]]:
        state_dict = {}

        for name, lr_scheduler in self.lr_schedulers.items():
            lr_scheduler: Optional[_LRScheduler]

            if lr_scheduler is None:
                state_dict[name] = {}
            else:
                state_dict[name] = lr_scheduler.state_dict(*args, **kwargs)

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        r"""Loads the learning rate scheduler state.

        Args:
            state_dict (dict): Learning rate scheduler state. Should be an object returned
                from a call to ``state_dict``.

        """

        for name, lr_scheduler in self.lr_schedulers.items():
            lr_scheduler: Optional[_LRScheduler]

            if lr_scheduler is None:
                assert len(state_dict[name]) == 0
            else:
                lr_scheduler.load_state_dict(state_dict[name])


class MultiLR(MultiLRSchedulers):
    """Alias of MultiLRSchedulers."""


class GANLRScheduler:
    def __init__(self, generator: _LRScheduler, discriminator: _LRScheduler) -> None:
        self.generator = generator
        self.discriminator = discriminator

    def step(self, *args, **kwargs) -> None:
        self.generator.step(*args, **kwargs)
        self.discriminator.step(*args, **kwargs)


class GANLR(GANLRScheduler):
    """Alias of GANLRScheduler."""
