import torch

from .vqvae import CodebookEntropyLoss as VQVAECodebookEntropyLoss
from .vqvae import CodebookUsageLoss as VQVAECodebookUsageLoss

__all__ = [
    "CodebookEntropyLoss",
    "CodebookUsageLoss",
]


class CodebookEntropyLoss(VQVAECodebookEntropyLoss):
    def forward(self, input: torch.LongTensor) -> torch.Tensor:
        """Forward pass of CodebookEntropyLoss.

        .. note::

            This loss is not differentiable, so use this for monitoring.

        Args:
            input (torch.LongTensor): Selected codebook indices of shape
                (batch_size, num_stages, *).

        Returns:
            torch.Tensor: Entropy of shape ().

        """
        input = input.transpose(1, 0).contiguous()
        num_stages = input.size(0)

        loss = 0

        for _input in input:
            loss = loss + super().forward(_input)

        loss = loss / num_stages

        return loss


class CodebookUsageLoss(VQVAECodebookUsageLoss):
    def forward(self, input: torch.LongTensor) -> torch.Tensor:
        """Forward pass of CodebookUsageLoss.

        .. note::

            This loss is not differentiable, so use this for monitoring.

        Args:
            input (torch.LongTensor): Selected codebook indices of shape
                (batch_size, num_stages, *).

        Returns:
            torch.Tensor: Usage of shape ().

        """
        input = input.transpose(1, 0).contiguous()
        num_stages = input.size(0)

        loss = 0

        for _input in input:
            loss = loss + super().forward(_input)

        loss = loss / num_stages

        return loss
