from typing import Optional

import torch
import torch.nn.functional as F

from audyn.criterion.contrastive import IntraInfoNCELoss as BaseIntraInfoNCELoss


class IntraInfoNCELoss(BaseIntraInfoNCELoss):
    def __init__(
        self,
        dim: int,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            dim,
            temperature=1,
            trainable=False,
            min_temperature=None,
            max_temperature=None,
            reduction=reduction,
        )

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of _InfoNCELoss.

        Args:
            input (torch.Tensor): Feature of shape (*, num_features).
            target (torch.Tensor): Feature of shape (*, num_features).

        Returns:
            torch.Tensor: Computed loss.

        """
        dim = self.dim
        n_dims = input.dim()

        self.validate_input_shape(input)
        self.validate_input_shape(target)

        if dim < 0:
            dim = n_dims + dim

        # permute dims of input and target
        input = self.permute_along_dim(input)
        target = self.permute_along_dim(target)
        target = target.transpose(-2, -1)

        sample_size = input.size(-2)
        logit = torch.matmul(input, target)
        indices = torch.arange(sample_size, device=logit.device)

        indices_size = input.size()[:-2] + (sample_size,)
        indices = indices.expand(*indices_size)
        loss = self.cross_entropy(logit, indices, length=length)

        return loss

    def cross_entropy(
        self,
        logit: torch.Tensor,
        target: torch.LongTensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Cross entropy from one view.

        Args:
            logit (torch.Tensor): Logit of shape (*, num_samples, num_samples).
            target (torch.LongTensor): Target indices of shape (*, num_samples).
            length (torch.LongTensor, optioinal): Target indices of shape (*,).

        Returns:
            torch.Tensor: Computed cross entropy from one view.

        """
        reduction = self.reduction
        ignore_idx = -1

        # permute dims of logit
        n_dims = logit.dim()
        dims = tuple(range(n_dims))
        left_dims = dims[:1]
        right_dims = dims[1:-1]
        sample_dim = dims[-1:]
        dims = left_dims + sample_dim + right_dims
        logit = logit.permute(*dims)
        num_samples = logit.size(1)

        indices = torch.arange(num_samples, device=logit.device)
        padding_mask = indices >= length.unsqueeze(dim=-1)

        logit = logit.masked_fill(padding_mask.unsqueeze(dim=-1), -float("inf"))
        target = target.masked_fill(padding_mask, ignore_idx)

        loss = F.cross_entropy(logit, target, reduction="none", ignore_index=ignore_idx)

        if reduction == "mean":
            loss = loss.sum(dim=-1) / length
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum(dim=-1)
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"{reduction} is not supported as reduction.")

        return loss
