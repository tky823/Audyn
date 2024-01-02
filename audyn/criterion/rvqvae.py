import torch

from .vqvae import CodebookEntropyLoss as VQVAECodebookEntropyLoss
from .vqvae import CodebookLoss as VQVAECodebookLoss
from .vqvae import CommitmentLoss as VQVAECommitmentLoss

__all__ = ["CodebookLoss", "CommitmentLoss"]


class CodebookLoss(VQVAECodebookLoss):
    """Codebook loss to update embeddings in codebook."""

    def __init__(self, stage_wise: bool = True, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)

        self.stage_wise = stage_wise

    def forward(self, encoded: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        """Forward pass of CodebookLoss.

        Args:
            encoded (torch.Tensor): Encoded feature. Any shape can be handled.
                This tensor is atached in loss computation.
            quantized (torch.Tensor): Quantized feature of same shape as encoded feature.

        Returns:
            torch.Tensor: Computed loss. The shape depends on ``reduction``.

        """
        if self.stage_wise:
            quantized = quantized.sum(dim=1)

        loss = super().forward(quantized, encoded)

        return loss


class CommitmentLoss(VQVAECommitmentLoss):
    """Codebook loss to update encoded feature."""

    def __init__(self, stage_wise: bool = True, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)

        self.stage_wise = stage_wise

    def forward(self, encoded: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        """Forward pass of CommitmentLoss.

        Args:
            encoded (torch.Tensor): Encoded feature. Any shape can be handled.
            quantized (torch.Tensor): Quantized feature of same shape as encoded feature.
                This tensor is atached in loss computation.

        Returns:
            torch.Tensor: Computed loss. The shape depends on ``reduction``.

        """
        if self.stage_wise:
            quantized = quantized.sum(dim=1)

        loss = super().forward(encoded, quantized)

        return loss


class CodebookEntropyLoss(VQVAECodebookEntropyLoss):
    def forward(self, input: torch.LongTensor) -> torch.Tensor:
        """Forward pass of CodebookEntropyLoss.

        .. note::

            This loss is not differentiable, so use this for monitoring.

        Args:
            input (torch.Tensor): Selected codebook indices of shape (batch_size, num_stages, *).

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
