from typing import Optional

import torch

__all__ = ["transform_log_duration", "to_long"]


def transform_log_duration(
    log_duration: torch.Tensor,
    min_duration: int = 1,
    max_duration: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Transform log-duration to linear-duration.

    Args:
        log_duration (torch.Tensor): Duration in log-domain.
        min_duration (int): Min duration in linear domain. Default: ``1``.
        max_duration (int, optional): Max duration in linear domain. Default: ``None``.
        dtype (torch.dtype, optional): Type of linear_duration.

    Returns:
        torch.Tensor: Duration in linear-domain.

    """
    linear_duration = torch.exp(log_duration)
    linear_duration = torch.round(linear_duration.detach())
    linear_duration = torch.clip(linear_duration, min=min_duration, max=max_duration)

    if dtype is not None:
        linear_duration = linear_duration.to(dtype)

    return linear_duration


def to_long(duration: torch.Tensor) -> torch.LongTensor:
    """Convert duration to long type."""
    return duration.long()
