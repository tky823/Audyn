import torch
import torch.nn as nn
from torch.optim import SGD

from audyn.optim.lr_scheduler import TransformerLRScheduler


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
