"""
Feature-wise Linear Modulation
    Reference: "FiLM: Visual Reasoning with a General Conditioning Layer"
    See https://arxiv.org/abs/1709.07871
"""

import torch
import torch.nn as nn

__all__ = [
    "FiLM",
    "FiLM1d",
    "FiLM2d",
]


class FiLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of FiLM.

        Args:
            input (torch.Tensor): (batch_size, num_features, *)
            gamma (torch.Tensor): (batch_size, num_features)
            beta (torch.Tensor): (batch_size, num_features)

        Returns:
            torch.Tensor: Output of same shape as input.

        """
        n_dims = input.dim()
        expand_dims = (1,) * (n_dims - 2)
        dims = gamma.size() + expand_dims

        gamma = gamma.view(*dims)
        beta = beta.view(*dims)

        return gamma * input + beta


class FiLM1d(FiLM):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of FiLM1d.

        Args:
            input (torch.Tensor): (batch_size, num_features, length)
            gamma (torch.Tensor): (batch_size, num_features)
            beta (torch.Tensor): (batch_size, num_features)

        Returns:
            torch.Tensor: Output of same shape as input.

        """
        dims = gamma.size() + (1,)

        gamma = gamma.view(*dims)
        beta = beta.view(*dims)

        return gamma * input + beta


class FiLM2d(FiLM):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, gamma, beta):
        """
        Args:
            input (torch.Tensor): (batch_size, num_features, height, width)
            gamma (torch.Tensor): (batch_size, num_features)
            beta (torch.Tensor): (batch_size, num_features)

        Returns:
            torch.Tensor: Output of same shape as input.

        """
        dims = gamma.size() + (1, 1)

        gamma = gamma.view(*dims)
        beta = beta.view(*dims)

        return gamma * input + beta
