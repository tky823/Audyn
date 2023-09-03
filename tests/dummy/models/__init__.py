import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear = nn.Linear(1, 1, bias=False)
        self.linear.weight.data.fill_(1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.linear(input)

        return output

    @torch.no_grad()
    def inference(self, input: torch.Tensor) -> torch.Tensor:
        output = self.linear(input)

        return output
