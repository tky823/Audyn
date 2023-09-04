from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["TransformerLRScheduler", "GANLRScheduler"]


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


class GANLRScheduler:
    def __init__(self, generator: _LRScheduler, discriminator: _LRScheduler) -> None:
        self.generator = generator
        self.discriminator = discriminator

    def step(self, *args, **kwargs) -> None:
        self.generator.step(*args, **kwargs)
        self.discriminator.step(*args, **kwargs)
