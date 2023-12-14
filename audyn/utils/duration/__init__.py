from typing import Optional

import torch

__all__ = ["transform_log_duration"]


def transform_log_duration(
    log_duration: torch.Tensor, min_duration: int = 1, max_duration: Optional[int] = None
) -> torch.Tensor:
    """Transform log-duration to linear-duration.

    Args:
        log_duration (torch.Tensor): Duration in log-domain.
        min_duration (int): Min duration in linear domain. Default: ``1``.
        max_duration (int, optional): Max duration in linear domain. Default: ``None``.

    Returns:
        torch.Tensor: Duration in linear-domain.

    """
    linear_duration = torch.exp(log_duration)
    linear_duration = torch.round(linear_duration.detach())
    linear_duration = torch.clip(linear_duration, min=min_duration, max=max_duration)

    return linear_duration
