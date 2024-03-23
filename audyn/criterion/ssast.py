from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "ReconstructionLoss",
    "ClassificationLoss",
    "SSASTReconstructionLoss",
    "SSASTClassificationLoss",
]


class ReconstructionLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        self.reduction = reduction

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, length: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        factory_kwargs = {
            "dtype": torch.long,
            "device": input.device,
        }
        batch_size, max_length, embedding_dim = input.size()

        if length is None:
            length = torch.full(
                (batch_size,),
                fill_value=max_length,
                *factory_kwargs,
            )

        padding_mask = torch.arange(max_length, **factory_kwargs) >= length.unsqueeze(dim=-1)
        loss = (input - target) ** 2
        loss = loss.masked_fill(padding_mask.unsqueeze(dim=-1), 0)

        if self.reduction == "mean":
            non_padding_mask = torch.logical_not(padding_mask)
            non_padding_mask = non_padding_mask.long()
            loss = loss.sum() / (embedding_dim * non_padding_mask.sum())
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"{self.reduction} is not supported as reduction.")

        return loss


class ClassificationLoss(nn.Module):
    """Classification loss.

    ref: audyn.criterion.contrastive.InfoNCELoss.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        self.reduction = reduction

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, length: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        factory_kwargs = {
            "dtype": torch.long,
            "device": input.device,
        }
        batch_size, max_length, _ = input.size()

        if length is None:
            length = torch.full(
                (batch_size,),
                fill_value=max_length,
                *factory_kwargs,
            )

        padding_mask = torch.arange(max_length, **factory_kwargs) >= length.unsqueeze(dim=-1)
        logit = torch.matmul(target, input.transpose(2, 1))
        logit_diag = torch.diagonal(logit, dim1=-2, dim2=-1)
        logit = logit.masked_fill(padding_mask.unsqueeze(dim=-2), -float("inf"))
        logsumexp = torch.logsumexp(logit, dim=-1)
        loss = -logit_diag + logsumexp
        loss = loss.masked_fill(padding_mask, 0)

        if self.reduction == "mean":
            non_padding_mask = torch.logical_not(padding_mask)
            non_padding_mask = non_padding_mask.to(torch.long)
            loss = loss.sum() / non_padding_mask.sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"{self.reduction} is not supported as reduction.")

        return loss


class SSASTReconstructionLoss(ReconstructionLoss):
    """Alias of ReconstructionLoss."""


class SSASTClassificationLoss(ClassificationLoss):
    """Alias of ClassificationLoss."""
