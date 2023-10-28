import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CodebookLoss", "CommitmentLoss"]


class CodebookLoss(nn.Module):
    """Codebook loss to update embeddings in codebook."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        self.reduction = reduction

    def forward(self, encoded: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        """Forward pass of CodebookLoss.

        Args:
            encoded (torch.Tensor): Encoded feature. Any shape can be handled.
                This tensor is atached in loss computation.
            quantized (torch.Tensor): Quantized feature of same shape as encoded feature.

        Returns:
            torch.Tensor: Computed loss. The shape depends on ``reduction``.

        """
        loss = vqvae_mse_loss(quantized, encoded, reduction=self.reduction)

        return loss


class CommitmentLoss(nn.Module):
    """Codebook loss to update encoded feature."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        self.reduction = reduction

    def forward(self, encoded: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        """Forward pass of CommitmentLoss.

        Args:
            encoded (torch.Tensor): Encoded feature. Any shape can be handled.
            quantized (torch.Tensor): Quantized feature of same shape as encoded feature.
                This tensor is atached in loss computation.

        Returns:
            torch.Tensor: Computed loss. The shape depends on ``reduction``.

        """
        loss = vqvae_mse_loss(encoded, quantized, reduction=self.reduction)

        return loss


def vqvae_mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "none",
) -> torch.Tensor:
    "MSE with input and detached target."
    loss = F.mse_loss(input, target.detach(), reduction=reduction)

    return loss
