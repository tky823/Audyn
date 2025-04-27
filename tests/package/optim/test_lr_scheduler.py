import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LinearLR, StepLR

from audyn.optim.lr_scheduler import (
    BurnInLRScheduler,
    MultiLRSchedulers,
    TransformerLRScheduler,
)
from audyn.optim.lr_scheduler import (
    ExponentialWarmupLinearCooldownLRScheduler as PaSSTLRScheduler,
)
from audyn.optim.optimizer import MultiOptimizers


def test_transformer_lr_scheduler() -> None:
    torch.manual_seed(0)

    class Model(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()

            self.linear = nn.Linear(in_channels, out_channels)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.linear(input)

    class Criterion(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.mean(input - target)

    batch_size = 4
    in_channels, out_channels = 3, 1
    lr = 0.1

    model = Model(in_channels, out_channels)
    optimizer = SGD(model.parameters(), lr=lr)
    lr_scheduler = TransformerLRScheduler(optimizer, in_channels)
    criterion = Criterion()

    input = torch.randn((batch_size, in_channels))
    target = torch.randn((batch_size, out_channels))

    output = model(input)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()


def test_burnin_lr_scheduler() -> None:
    torch.manual_seed(0)

    class Model(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()

            self.linear = nn.Linear(in_channels, out_channels)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.linear(input)

    class Criterion(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.mean(input - target)

    batch_size = 4
    in_channels, out_channels = 3, 1
    lr = 0.1
    burnin_step, burnin_scale = 3, 0.01

    model = Model(in_channels, out_channels)
    optimizer = SGD(model.parameters(), lr=lr)
    lr_scheduler = BurnInLRScheduler(optimizer, burnin_step=burnin_step, burnin_scale=burnin_scale)
    criterion = Criterion()

    for iteration_idx in range(5):
        input = torch.randn((batch_size, in_channels))
        target = torch.randn((batch_size, out_channels))

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        for param_group in optimizer.param_groups:
            if iteration_idx + 1 < burnin_step:
                assert param_group["lr"] == lr * burnin_scale
            else:
                assert param_group["lr"] == lr, iteration_idx


@pytest.mark.parametrize(
    "optim_type",
    [
        "list_dict",
        "list_optim",
    ],
)
def test_multi_optimizers(optim_type: str) -> None:
    torch.manual_seed(0)

    class CustomModel(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()

            self.linear = nn.Linear(in_channels, out_channels)
            self.norm = nn.LayerNorm(out_channels)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                input (torch.Tensor): Tensor of shape (*, in_channels).

            Returns:
                torch.Tensor: Tensor of shape (*, out_channels).

            """
            x = self.linear(input)
            output = self.norm(x)

            return output

    batch_size = 2
    in_channels, out_channels = 3, 5
    iterations = 10

    model = CustomModel(in_channels, out_channels)
    sgd_optimizer = SGD(model.linear.parameters(), lr=0.1)
    adam_optimizer = Adam(model.norm.parameters(), lr=0.01)

    if optim_type == "list_dict":
        optimizers = [
            {
                "name": "sgd",
                "optimizer": sgd_optimizer,
            },
            {
                "name": "adam",
                "optimizer": adam_optimizer,
            },
        ]
    elif optim_type == "list_optim":
        optimizers = [
            sgd_optimizer,
            adam_optimizer,
        ]
    else:
        raise ValueError(f"{type(optim_type)} is not suppored as optim_type.")

    step_lr_scheduler = StepLR(sgd_optimizer, 2, gamma=0.9)
    linear_lr_scheduler = LinearLR(adam_optimizer, 1, 0.5, total_iters=iterations)

    if optim_type == "list_dict":
        lr_schedulers = [
            {
                "name": "step",
                "lr_scheduler": step_lr_scheduler,
            },
            {
                "name": "linear",
                "lr_scheduler": linear_lr_scheduler,
            },
        ]
    elif optim_type == "list_optim":
        lr_schedulers = [
            step_lr_scheduler,
            linear_lr_scheduler,
        ]
    else:
        raise ValueError(f"{type(optim_type)} is not suppored as optim_type.")

    optimizer = MultiOptimizers(optimizers)
    lr_scheduler = MultiLRSchedulers(lr_schedulers)

    for _ in range(iterations):
        input = torch.randn((batch_size, in_channels))
        output = model(input)
        loss = torch.mean(output)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        optimizer_state_dict = optimizer.state_dict()
        lr_scheduler_state_dict = lr_scheduler.state_dict()

        optimizer.load_state_dict(optimizer_state_dict)
        lr_scheduler.load_state_dict(lr_scheduler_state_dict)


def test_passt_lr_scheduler() -> None:
    torch.manual_seed(0)

    batch_size = 2
    in_channels, out_channels = 3, 5
    iterations = 10  # update twice

    model = nn.Linear(in_channels, out_channels)
    optimizer = SGD(model.parameters(), lr=0.1)
    lr_scheduler = PaSSTLRScheduler(optimizer, warmup_steps=2, suspend_steps=1, cooldown_steps=3)

    for _ in range(iterations):
        input = torch.randn((batch_size, in_channels))
        output = model(input)
        loss = torch.mean(output)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    state_dict = {
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }

    optimizer.load_state_dict(state_dict["optimizer"])
    lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
