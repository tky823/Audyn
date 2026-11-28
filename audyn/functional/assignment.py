import os

import torch

__all__ = [
    "optimal_assignment",
]


@torch.no_grad()
def optimal_assignment(cost: torch.Tensor) -> torch.LongTensor:
    """Find minimum-cost assignment for square cost matrices.

    Args:
        cost (torch.Tensor): Cost matrix of shape (num_rows, num_rows) or cost matrices
            of shape (batch_size, num_rows, num_rows). The dtype must be ``torch.float32``.

    Returns:
        torch.LongTensor: Column index for each row of shape (num_rows,) or
            (batch_size, num_rows). The output is on the same device as ``cost``.

    .. note::

        Assignment is always computed on CPU.

    """
    if cost.dim() not in [2, 3]:
        raise ValueError("cost must be a 2- or 3-dimensional tensor.")

    if cost.size(-2) != cost.size(-1):
        raise ValueError("cost must contain square matrices.")

    if cost.dtype != torch.float32:
        raise ValueError("cost must have float32 dtype.")

    is_batched = cost.dim() == 3
    src_device = cost.device
    cost = cost.cpu()

    if not is_batched:
        cost = cost.unsqueeze(dim=0)

    cost = cost.contiguous()
    hungarian = _load_hungarian()
    assignment = hungarian.hungarian(cost)

    if not is_batched:
        assignment = assignment.squeeze(dim=0)

    assignment = assignment.to(src_device)

    return assignment


def _load_hungarian():
    try:
        from .._C import hungarian
    except ImportError:
        from torch.utils.cpp_extension import load

        source_dir = os.path.join(os.path.dirname(__file__), "..", "..", "csrc")

        return load(name="audyn._C.hungarian", sources=[os.path.join(source_dir, "hungarian.cpp")])

    return hungarian
