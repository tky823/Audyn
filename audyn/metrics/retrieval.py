from typing import List, Optional, Union

import torch

from .base import StatefulMetric

__all__ = ["MeanAveragePrecision"]


class MeanAveragePrecision(StatefulMetric):
    """Mean average precision (mAP).

    Args:
        k (int): Threshold of retrieval rank. This parameter should be positive. For example,
            to evaluate mAP@5, set 5 as ``k``.
        mink (int): Start index of rank. This is useful to specify top rank is 0 or 1.
        enforce_sorted (bool): If ``True``, rank is sorted for ``update``. Default: ``False``.

    """

    def __init__(
        self,
        k: int,
        mink: int = 0,
        enforce_sorted: bool = False,
    ) -> None:
        super().__init__()

        assert k > 0, "k should be positive."

        self.k = k
        self.mink = mink
        self.enforce_sorted = enforce_sorted

        self.reset()

    def reset(self) -> None:
        """Reset ``num_samples`` and ``sum_ap``."""
        self.num_samples = 0
        self.sum_ap = 0

    def update(
        self,
        ranks: Union[int, List[int], torch.LongTensor],
        enforce_sorted: Optional[bool] = None,
    ) -> None:
        """Update state of MeanAveragePrecision.

        Args:
            ranks (int, list, or torch.LongTensor): Retrieved ranks. Number of items
                corresponds to groundtruth recommendation items. If ``ranks`` is int,
                Number of items is regarded as 1.
            enforce_sorted (bool, optional): If ``True``, rank is sorted. By default,
                ``self.enforce_sorted`` is used.

        .. note::

            In this function, minimum of ``ranks`` is assumed to be ``self.mink``.

        """
        k = self.k

        if enforce_sorted is None:
            enforce_sorted = self.enforce_sorted

        if isinstance(ranks, int):
            ranks = [ranks]
        elif isinstance(ranks, list):
            pass
        elif isinstance(ranks, torch.Tensor):
            ranks = ranks.tolist()
        else:
            raise ValueError(f"Invalid type {type(ranks)} is given as ranks.")

        self.num_samples = self.num_samples + 1
        ap = self._compute_average_precision(ranks, enforce_sorted=enforce_sorted)
        self.sum_ap = self.sum_ap + ap / min(k, len(ranks))

    def compute(self) -> torch.Tensor:
        return torch.tensor(self.sum_ap / self.num_samples)

    def _compute_average_precision(
        self, rank: List[int], enforce_sorted: Optional[bool] = False
    ) -> float:
        k = self.k
        mink = self.mink

        if enforce_sorted:
            rank = sorted(rank)

        ap_at_k = 0

        for tp_k, _rank in enumerate(rank[:k], 1):
            if _rank - mink >= k:
                break

            ap_at_k = ap_at_k + tp_k / (_rank - mink + 1)

        return ap_at_k
