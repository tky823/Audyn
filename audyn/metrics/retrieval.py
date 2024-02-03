from typing import List, Optional, Union

import torch
import torch.distributed as dist

from .base import StatefulMetric

__all__ = ["MeanAveragePrecision", "MedianRank"]


class MeanAveragePrecision(StatefulMetric):
    """Mean average precision (mAP).

    .. note::

        This class supports distributed data parallel.

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
        device: torch.device = None,
    ) -> None:
        super().__init__(device=device)

        assert k > 0, "k should be positive."

        self.k = k
        self.mink = mink
        self.enforce_sorted = enforce_sorted

        self.reset()

    def reset(self) -> None:
        """Reset ``num_samples`` and ``sum_ap``."""
        self.num_samples = 0
        self.sum_ap = 0

    @torch.no_grad()
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

        is_distributed = dist.is_available() and dist.is_initialized()

        if is_distributed:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        if isinstance(ranks, int):
            ranks = [ranks]
        elif isinstance(ranks, list):
            pass
        elif isinstance(ranks, torch.Tensor):
            ranks = ranks.tolist()
        else:
            raise ValueError(f"Invalid type {type(ranks)} is given as ranks.")

        ap = self._compute_average_precision(ranks, enforce_sorted=enforce_sorted)
        normalized_ap = ap / min(k, len(ranks))

        if is_distributed:
            tensor = torch.tensor(normalized_ap, device=self.device)
            gathered_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered_tensor, tensor)
            gathered_tensor = torch.stack(gathered_tensor, dim=0)
            gathered_tensor = gathered_tensor.sum(dim=0)
            normalized_ap = gathered_tensor.item()

        self.num_samples = self.num_samples + world_size
        self.sum_ap = self.sum_ap + normalized_ap

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        map_k = torch.tensor(self.sum_ap / self.num_samples)

        return map_k.to(self.device)

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


class MedianRank(StatefulMetric):
    """Median rank (medR).

    .. note::

        This class supports distributed data parallel.

    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)

        self.reset()

    def reset(self) -> None:
        self.ranks = []

    @torch.no_grad()
    def update(self, rank: Union[int, torch.Tensor]) -> None:
        is_distributed = dist.is_available() and dist.is_initialized()

        if is_distributed:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        if isinstance(rank, int):
            pass
        elif isinstance(rank, torch.Tensor):
            assert rank.numel() == 1, "Only scalar is supported as rank."

            rank = rank.detach().item()
        else:
            raise ValueError(f"{type(rank)} is not supported as rank.")

        if is_distributed:
            tensor = torch.tensor(rank, device=self.device)
            gathered_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered_tensor, tensor)
            gathered_tensor = torch.stack(gathered_tensor, dim=0)
            ranks = gathered_tensor.tolist()
            self.ranks.extend(ranks)
        else:
            self.ranks.append(rank)

    @torch.no_grad()
    def compute(self) -> torch.LongTensor:
        ranks = torch.tensor(self.ranks)
        med = torch.median(ranks)

        return med.to(self.device)
