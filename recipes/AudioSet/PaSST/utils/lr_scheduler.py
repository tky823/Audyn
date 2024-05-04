import math


class ExponentialWarmupLinearCooldownLambda:
    """Exponential warm-up + linear cool-down of learning rate.

    Args:
        warmup_steps (int): Number of exponential warm-up steps.
        constant_steps (int): Number of constant learning rate steps between warm-up and cool-down.
        cooldown_steps (int): Number of linear cool-down steps after constant learning rate.
        last_factor (float): Scale factor of learning rate at last step.

    """

    def __init__(
        self,
        warmup_steps: int,
        constant_steps: int,
        cooldown_steps: int,
        last_factor: float = 1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.cooldown_steps = cooldown_steps
        self.last_factor = last_factor

    def __call__(self, step: int) -> float:
        warmup_steps = self.warmup_steps
        constant_steps = self.constant_steps
        cooldown_steps = self.cooldown_steps
        last_factor = self.last_factor

        if step < warmup_steps:
            step = min(step, warmup_steps)
            normalized_step = 1 - step / warmup_steps
            factor = math.exp(-5.0 * normalized_step**2)
        elif step < warmup_steps + constant_steps:
            factor = 1
        elif step < warmup_steps + constant_steps + cooldown_steps:
            step_after_constant = step - (warmup_steps + constant_steps)
            normalized_step = step_after_constant / cooldown_steps
            factor = last_factor + (1 - last_factor) * normalized_step
        else:
            factor = last_factor

        return factor
