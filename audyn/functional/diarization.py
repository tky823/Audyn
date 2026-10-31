import math
from itertools import combinations
from typing import overload

import torch

__all__ = [
    "powerset_to_multilabel",
]


@overload
def powerset_to_multilabel(
    input: torch.Tensor, num_classes: int, max_subset: int
) -> torch.Tensor: ...


@overload
def powerset_to_multilabel(
    input: torch.Tensor, powerset_to_multilabel: torch.Tensor
) -> torch.Tensor: ...


def powerset_to_multilabel(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Convert powerset scores to frame-level binary class activity.

    For each frame, the highest-scoring powerset class is selected. Powerset
    classes are ordered by subset size, then lexicographically within each size;
    the empty subset is first.

    Args:
        input (torch.Tensor): Scores of shape
            ``(batch_size, num_powerset_classes, num_frames)``.
        num_classes (int, optional): Number of output classes. Specify together
            with ``max_subset`` when a lookup table is not provided.
        max_subset (int, optional): Maximum number of active classes in one
            powerset class. Specify together with ``num_classes``.
        powerset_to_multilabel (torch.Tensor, optional): Precomputed binary lookup
            table of shape ``(num_powerset_classes, num_classes)`` on the same
            device as ``input``. Specify this instead of ``num_classes`` and
            ``max_subset``. These arguments can be passed positionally or by name.

    Returns:
        torch.Tensor: Binary activity of shape
            ``(batch_size, num_classes, num_frames)``. Its dtype is ``float32``
            when the lookup table is built here, or the supplied table's dtype.

    """
    if input.dim() != 3:
        raise ValueError(f"Expected 3-dimensional input, but got {input.dim()} dimensions.")

    build_powerset_to_multilabel = False

    if len(args) == 1 and not kwargs:
        _powerset_to_multilabel = args[0]
    elif not args and set(kwargs) == {"powerset_to_multilabel"}:
        _powerset_to_multilabel = kwargs["powerset_to_multilabel"]
    elif len(args) == 2 and not kwargs:
        num_classes, max_subset = args
        build_powerset_to_multilabel = True
    elif len(args) == 1 and set(kwargs) == {"max_subset"}:
        num_classes, max_subset = args[0], kwargs["max_subset"]
        build_powerset_to_multilabel = True
    elif not args and set(kwargs) == {"num_classes", "max_subset"}:
        num_classes, max_subset = kwargs["num_classes"], kwargs["max_subset"]
        build_powerset_to_multilabel = True
    else:
        raise ValueError(
            "Specify either powerset_to_multilabel, or both num_classes and max_subset."
        )

    if build_powerset_to_multilabel:
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer.")

        if not isinstance(max_subset, int) or not 0 <= max_subset <= num_classes:
            raise ValueError("max_subset must be an integer between 0 and num_classes.")

        num_powerset_classes = sum(math.comb(num_classes, size) for size in range(max_subset + 1))
        _powerset_to_multilabel = torch.zeros(
            num_powerset_classes, num_classes, dtype=torch.float32, device=input.device
        )
        index = 0

        for size in range(max_subset + 1):
            for subset in combinations(range(num_classes), size):
                _powerset_to_multilabel[index, list(subset)] = 1
                index += 1
    else:
        if (
            not isinstance(_powerset_to_multilabel, torch.Tensor)
            or _powerset_to_multilabel.dim() != 2
        ):
            raise ValueError("powerset_to_multilabel must be a 2-dimensional tensor.")

        if _powerset_to_multilabel.device != input.device:
            raise ValueError("powerset_to_multilabel must be on the same device as input.")

    num_powerset_classes = _powerset_to_multilabel.size(0)

    if input.size(1) != num_powerset_classes:
        raise ValueError(
            f"Expected {num_powerset_classes} powerset classes, but got {input.size(1)}."
        )

    indices = torch.argmax(input, dim=1)
    multilabel = _powerset_to_multilabel[indices]
    output = multilabel.transpose(1, 2).contiguous()

    return output
