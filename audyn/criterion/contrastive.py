import math
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "InfoNCELoss",
    "NTXentLoss",
    "IntraInfoNCELoss",
    "InterInfoNCELoss",
    "IntraNTXentLoss",
    "InterNTXentLoss",
]


class _ContrastiveLoss(nn.Module):
    """Base class of contrastive loss.

    NOTE: This class is extended for InfoNCE and NT-Xent losses.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        temperature: float = 1,
        trainable: bool = True,
        min_temperature: Optional[float] = None,
        max_temperature: Optional[float] = None,
        reduction: str = "mean",
        gather_in_eval: Optional[bool] = None,
    ) -> None:
        super().__init__()

        log_temperature = math.log(temperature)
        self.log_temperature = nn.Parameter(
            torch.tensor(log_temperature),
            requires_grad=trainable,
        )

        self.dim = dim
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.reduction = reduction

        self.gather_if_necessary = None
        self.gather_in_eval = gather_in_eval

    def validate_input_shape(self, input: torch.Tensor) -> int:
        dim = self.dim
        n_dims = input.dim()

        assert n_dims >= 2

        if dim < 0:
            _dim = n_dims + dim
        else:
            _dim = dim

        if _dim == n_dims - 1:
            raise ValueError(f"Last dimension ({dim}) should not be used as dim.")

    @staticmethod
    def normalize_feature(input: torch.Tensor) -> torch.Tensor:
        output = F.normalize(input, p=2, dim=-1)

        return output

    def permute_along_dim(self, input: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        n_dims = input.dim()

        if dim < 0:
            dim = n_dims + dim
        else:
            dim = dim

        # permute dims of input and other
        dims = tuple(range(n_dims))
        left_dims = dims[:dim]
        sample_dim = dims[dim : dim + 1]
        right_dims = dims[dim + 1 : -1]
        feature_dim = dims[-1:]
        dims = left_dims + right_dims + sample_dim + feature_dim
        output = input.permute(*dims)

        return output

    def clamp_log_temperature(self, log_temperature: torch.Tensor) -> torch.Tensor:
        min_temperature = self.min_temperature
        max_temperature = self.max_temperature

        clamp_kwargs = {}

        if min_temperature is not None:
            clamp_kwargs["min"] = math.log(min_temperature)

        if max_temperature is not None:
            clamp_kwargs["max"] = math.log(max_temperature)

        if len(clamp_kwargs) > 0:
            log_temperature = torch.clamp(log_temperature, **clamp_kwargs)

        return log_temperature


class _InfoNCELoss(_ContrastiveLoss):
    """Base class of InfoNCE loss."""

    def __init__(
        self,
        dim: Optional[int] = None,
        temperature: float = 1,
        trainable: bool = True,
        min_temperature: Optional[float] = None,
        max_temperature: Optional[float] = None,
        reduction: str = "mean",
        gather_in_eval: Optional[bool] = None,
    ) -> None:
        super().__init__(
            dim=dim,
            temperature=temperature,
            trainable=trainable,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            reduction=reduction,
            gather_in_eval=gather_in_eval,
        )

    def permute_along_dim(self, input: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        n_dims = input.dim()

        if dim < 0:
            dim = n_dims + dim
        else:
            dim = dim

        # permute dims of input and other
        dims = tuple(range(n_dims))
        left_dims = dims[:dim]
        sample_dim = dims[dim : dim + 1]
        right_dims = dims[dim + 1 : -1]
        feature_dim = dims[-1:]
        dims = left_dims + right_dims + sample_dim + feature_dim
        output = input.permute(*dims)

        return output

    def cross_entropy(self, logit: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """Cross entropy from one view.

        Args:
            logit (torch.Tensor): Logit of shape (*, num_samples', num_samples).
            target (torch.LongTensor): Target indices of shape (*, num_samples).

        Returns:
            torch.Tensor: Computed cross entropy from one view.

        """
        reduction = self.reduction

        # permute dims of logit
        n_dims = logit.dim()
        dims = tuple(range(n_dims))
        left_dims = dims[:1]
        right_dims = dims[1:-1]
        sample_dim = dims[-1:]
        dims = left_dims + sample_dim + right_dims
        logit = logit.permute(*dims)

        loss = F.cross_entropy(logit, target, reduction=reduction)

        return loss

    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Forward pass of _InfoNCELoss.

        Args:
            input (torch.Tensor): Feature of shape (*, num_features).
            other (torch.Tensor): Feature of shape (*, num_features).

        Returns:
            torch.Tensor: Computed loss.

        """
        gather_if_necessary = self.gather_if_necessary
        dim = self.dim
        n_dims = input.dim()

        is_distributed = dist.is_available() and dist.is_initialized()

        if is_distributed:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        if world_size > 1 and gather_if_necessary:
            if self.training or self.gather_in_eval:
                should_gather = True
            else:
                should_gather = False
        else:
            should_gather = False

        self.validate_input_shape(input)

        if dim < 0:
            dim = n_dims + dim

        input = self.normalize_feature(input)
        other = self.normalize_feature(other)
        log_temperature = self.clamp_log_temperature(self.log_temperature)
        temperature = torch.exp(-log_temperature)  # NOTE: inversion

        # permute dims of input and other
        input = self.permute_along_dim(input)
        other = self.permute_along_dim(other)

        if should_gather:
            # gather input and other along dim
            input = SyncFunction.apply(input, -2)
            other = SyncFunction.apply(other, -2)

        sample_size = input.size(-2)
        logit = torch.matmul(input, other.transpose(-2, -1)) * temperature
        target = torch.arange(sample_size, device=logit.device)

        target_size = input.size()[:-2] + (sample_size,)
        target = target.expand(*target_size)

        loss_one = self.cross_entropy(logit, target)
        loss_other = self.cross_entropy(logit.transpose(-2, -1), target)
        loss = 0.5 * (loss_one + loss_other)

        return loss


class _NTXentLoss(_ContrastiveLoss):
    """Base class of NT-Xent loss (normalized temperature cross entropy loss)."""

    def __init__(
        self,
        dim: Optional[int] = None,
        temperature: float = 1,
        trainable: bool = True,
        min_temperature: Optional[float] = None,
        max_temperature: Optional[float] = None,
        reduction: str = "mean",
        gather_in_eval: Optional[bool] = None,
    ) -> None:
        super().__init__(
            dim=dim,
            temperature=temperature,
            trainable=trainable,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            reduction=reduction,
            gather_in_eval=gather_in_eval,
        )

    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Forward pass of _NTXentLoss.

        Args:
            input (torch.Tensor): Feature of shape (*, num_features).
            other (torch.Tensor): Feature of shape (*, num_features).

        Returns:
            torch.Tensor: Computed loss.

        """
        gather_if_necessary = self.gather_if_necessary
        reduction = self.reduction
        dim = self.dim
        n_dims = input.dim()

        is_distributed = dist.is_available() and dist.is_initialized()

        if is_distributed:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        if world_size > 1 and gather_if_necessary:
            if self.training or self.gather_in_eval:
                should_gather = True
            else:
                should_gather = False
        else:
            should_gather = False

        self.validate_input_shape(input)

        if dim < 0:
            dim = n_dims + dim

        input = self.normalize_feature(input)
        other = self.normalize_feature(other)
        log_temperature = self.clamp_log_temperature(self.log_temperature)
        temperature = torch.exp(-log_temperature)  # NOTE: inversion

        # permute dims of input and other
        input = self.permute_along_dim(input)
        other = self.permute_along_dim(other)

        if should_gather:
            # gather input and other along dim
            input = SyncFunction.apply(input, -2)
            other = SyncFunction.apply(other, -2)

        sample_size = input.size(-2)
        logit = torch.matmul(input, other.transpose(-2, -1)) * temperature
        padding_mask = torch.eye(sample_size, dtype=torch.bool, device=logit.device)
        logit_no_diag = logit.masked_fill(padding_mask, -float("inf"))
        logit_no_diag = torch.cat([logit_no_diag, logit_no_diag.transpose(-2, -1)], dim=-1)
        logit_diag = torch.diagonal(logit, dim1=-2, dim2=-1)
        logit_no_diag = torch.logsumexp(logit_no_diag, dim=-1)
        loss = -logit_diag + logit_no_diag

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction ({reduction}) is specified.")

        return loss


class InfoNCELoss(_InfoNCELoss):
    """InfoNSE loss."""

    def __init__(
        self,
        dim: int,
        temperature: float = 1,
        trainable: bool = True,
        min_temperature: Optional[float] = None,
        max_temperature: Optional[float] = None,
        reduction: str = "mean",
        gather_in_eval: Optional[bool] = None,
    ) -> None:
        super().__init__(
            dim,
            temperature=temperature,
            trainable=trainable,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            reduction=reduction,
            gather_in_eval=gather_in_eval,
        )

        assert dim >= 0, f"dim ({dim}) should be non-negative."

        if dim == 0:
            self.gather_if_necessary = True
        else:
            self.gather_if_necessary = False

        if self.gather_in_eval is None:
            self.gather_in_eval = self.gather_if_necessary


class NTXentLoss(_NTXentLoss):
    """NTXent loss."""

    def __init__(
        self,
        dim: int,
        temperature: float = 1,
        trainable: bool = True,
        min_temperature: Optional[float] = None,
        max_temperature: Optional[float] = None,
        reduction: str = "mean",
        gather_in_eval: Optional[bool] = None,
    ) -> None:
        super().__init__(
            dim,
            temperature=temperature,
            trainable=trainable,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            reduction=reduction,
            gather_in_eval=gather_in_eval,
        )

        assert dim >= 0, f"dim ({dim}) should be non-negative."

        if dim == 0:
            self.gather_if_necessary = True
        else:
            self.gather_if_necessary = False

        if self.gather_in_eval is None:
            self.gather_in_eval = self.gather_if_necessary


class IntraInfoNCELoss(_InfoNCELoss):
    """InfoNSE loss where samples are not gathered among GPUs."""

    def __init__(
        self,
        dim: int,
        temperature: float = 1,
        trainable: bool = True,
        min_temperature: Optional[float] = None,
        max_temperature: Optional[float] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            dim,
            temperature=temperature,
            trainable=trainable,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            reduction=reduction,
            gather_in_eval=False,
        )

        self.gather_if_necessary = False


class InterInfoNCELoss(_InfoNCELoss):
    """InfoNSE loss where samples are gathered among GPUs."""

    def __init__(
        self,
        dim: int,
        temperature: float = 1,
        trainable: bool = True,
        min_temperature: Optional[float] = None,
        max_temperature: Optional[float] = None,
        reduction: str = "mean",
        gather_in_eval: Optional[bool] = True,
    ) -> None:
        super().__init__(
            dim,
            temperature=temperature,
            trainable=trainable,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            reduction=reduction,
            gather_in_eval=gather_in_eval,
        )

        self.gather_if_necessary = True


class IntraNTXentLoss(_NTXentLoss):
    """NTXent loss where samples are not gathered among GPUs."""

    def __init__(
        self,
        dim: int,
        temperature: float = 1,
        trainable: bool = True,
        min_temperature: Optional[float] = None,
        max_temperature: Optional[float] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            dim,
            temperature=temperature,
            trainable=trainable,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            reduction=reduction,
            gather_in_eval=False,
        )

        self.gather_if_necessary = False


class InterNTXentLoss(_NTXentLoss):
    """NTXent loss where samples are gathered among GPUs."""

    def __init__(
        self,
        dim: int,
        temperature: float = 1,
        trainable: bool = True,
        min_temperature: Optional[float] = None,
        max_temperature: Optional[float] = None,
        reduction: str = "mean",
        gather_in_eval: bool = True,
    ) -> None:
        super().__init__(
            dim,
            temperature=temperature,
            trainable=trainable,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            reduction=reduction,
            gather_in_eval=gather_in_eval,
        )

        self.gather_if_necessary = True


class SyncFunction(torch.autograd.Function):
    # TODO: improve design
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        world_size = dist.get_world_size()

        ctx.dim = dim
        ctx.samples_size_per_device = tensor.size(dim)
        ctx.rank = dist.get_rank()

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, dim=dim)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, gathered_grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        dim = ctx.dim
        samples_size_per_device = ctx.samples_size_per_device
        rank = ctx.rank

        gathered_grad_input = gathered_grad_output.clone()
        dist.all_reduce(gathered_grad_input, op=dist.ReduceOp.SUM)

        num_total_samples = gathered_grad_input.size(dim)
        sections = [
            rank * samples_size_per_device,
            samples_size_per_device,
            num_total_samples - (rank + 1) * samples_size_per_device,
        ]

        _, gathered_grad_input, _ = torch.split(
            gathered_grad_input,
            sections,
            dim=dim,
        )

        return gathered_grad_input, None
