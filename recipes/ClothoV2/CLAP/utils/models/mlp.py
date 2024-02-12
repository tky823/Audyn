import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int) -> None:
        """Multi-layer perceptron."""
        super().__init__()

        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of two-layer MLP.

        Args:
            input (torch.Tensor): Input feature of shape (*, in_channels).

        Returns:
            torch.Tensor: Output feature of shape (*, out_channels).

        """
        x = self.linear1(input)
        x = self.relu(x)
        output = self.linear2(x)

        return output
