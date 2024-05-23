from typing import Optional

import torch

__all__ = [
    "round_clip",
]


def round_clip(
    input: torch.Tensor,
    min: Optional[float] = None,
    max: Optional[float] = None,
) -> torch.Tensor:
    """Differntiable round + clip used in BitNet.

    .. note::

        Gradient is given by straight through estimator.

    """
    kwargs = {}

    if min is not None:
        kwargs["min"] = min

    if max is not None:
        kwargs["max"] = max

    x = torch.round(input)

    if len(kwargs) > 0:
        x = torch.clamp(x, **kwargs)

    output = torch.detach(x - input) + input

    return output
