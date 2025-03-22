from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class GumbelVQVAELR(LambdaLR):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        start_factor: float,
        end_factor: float,
        last_epoch=-1,
        **kwargs,
    ) -> None:
        assert warmup_steps < total_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                normalized_step = step / warmup_steps
                factor = normalized_step + (1 - normalized_step) * start_factor
            else:
                normalized_step = (step - warmup_steps) / (total_steps - warmup_steps)
                factor = end_factor + (1 - normalized_step) * normalized_step

            return factor

        super().__init__(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch, **kwargs)
