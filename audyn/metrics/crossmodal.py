from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .retrieval import MeanAveragePrecision, MedianRank

__all__ = ["CrossModalEmbeddingMeanAveragePrecision", "CrossModalEmbeddingMedianRank"]


class CrossModalEmbeddingMeanAveragePrecision(MeanAveragePrecision):
    """Mean average precision (mAP) for cross-modal embeddings.

    .. note::

        This class supports distributed data parallel.

    Args:
        k (int): Threshold of retrieval rank. This parameter should be positive. For example,
            to evaluate mAP@5, set 5 as ``k``.
        enforce_sorted (bool): If ``True``, rank is sorted for ``update``. Default: ``False``.

    """

    def __init__(
        self,
        k: int,
    ) -> None:
        # NOTE: Top rank is 0 in torch.
        super().__init__(k, mink=0, enforce_sorted=False)

    @torch.no_grad()
    def update(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        index: torch.LongTensor,
    ) -> None:
        """Update states of CrossModalEmbeddingMeanAveragePrecision.

        Args:
            query (torch.Tensor): Query embedding of shape (embedding_dim,)
                or (batch_size, embedding_dim).
            key (torch.Tensor): Key embeddings to search of shape (num_samples, embedding_dim),
                where num_samples is number of groudtruth samples.
            index (torch.LongTensor): Target index of query in key of shape () or (batch_size,).

        """
        mink = self.mink
        embedding_dim = query.size(-1)
        is_distributed = dist.is_available() and dist.is_initialized()

        if is_distributed:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        # ensure batch dimensions of query and index
        num_samples = key.size(0)
        query = query.view(-1, embedding_dim)
        index = index.view(-1)

        if is_distributed:
            gathered_query = [torch.zeros_like(query) for _ in range(world_size)]
            gathered_key = [torch.zeros_like(key) for _ in range(world_size)]
            gathered_index = [torch.zeros_like(index) for _ in range(world_size)]
            dist.all_gather(gathered_query, query)
            dist.all_gather(gathered_key, key)
            dist.all_gather(gathered_index, index)
            gathered_query = torch.stack(gathered_query, dim=0)
            gathered_key = torch.stack(gathered_key, dim=0)
            gathered_index = torch.stack(gathered_index, dim=0)
            query = gathered_query.view(-1, embedding_dim)
            key = gathered_key.view(-1, embedding_dim)
            offset = torch.arange(0, world_size * num_samples, num_samples, device=index.device)
            gathered_index = gathered_index + offset.unsqueeze(dim=-1)
            index = gathered_index.view(-1)

        _validate_shapes(query, key, index)

        similarity = F.cosine_similarity(query.unsqueeze(dim=-2), key, dim=-1)
        num_samples = similarity.size(-1)
        similarity = similarity.view(-1, num_samples)
        index = index.view(-1)

        _, retrieved_indices = torch.topk(similarity, k=num_samples, dim=-1)

        for _retrieved_indices, _idx in zip(retrieved_indices, index):
            _retrieved_indices = _retrieved_indices.tolist()
            _idx = _idx.item()
            _retrieved_rank = _retrieved_indices.index(_idx)

            super().update(_retrieved_rank + mink)


class CrossModalEmbeddingMedianRank(MedianRank):
    """Median rank for cross-modal embeddings.

    .. note::

        This class supports distributed data parallel.

    Args:
        mink (int): Start index of rank. This is useful to specify top rank is 0 or 1.

    """

    def __init__(self, mink: int = 0, device: Optional[torch.device] = None) -> None:
        super().__init__(device=device)

        self.mink = mink

    @torch.no_grad()
    def update(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        index: torch.LongTensor,
    ) -> None:
        """Update states of CrossModalEmbeddingMedianRank.

        Args:
            query (torch.Tensor): Query embedding of shape (embedding_dim,)
                or (batch_size, embedding_dim).
            key (torch.Tensor): Key embeddings to search of shape (num_samples, embedding_dim),
                where num_samples is number of groudtruth samples.
            index (torch.LongTensor): Target index of query in key of shape () or (batch_size,).

        .. note::

            In this implementation, index indicates target embeddings of ``key`` in same device
            as ``query``.

        """
        mink = self.mink
        embedding_dim = query.size(-1)
        is_distributed = dist.is_available() and dist.is_initialized()

        if is_distributed:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        # ensure batch dimensions of query and index
        num_samples = key.size(0)
        query = query.view(-1, embedding_dim)
        index = index.view(-1)

        if is_distributed:
            gathered_query = [torch.zeros_like(query) for _ in range(world_size)]
            gathered_key = [torch.zeros_like(key) for _ in range(world_size)]
            gathered_index = [torch.zeros_like(index) for _ in range(world_size)]
            dist.all_gather(gathered_query, query)
            dist.all_gather(gathered_key, key)
            dist.all_gather(gathered_index, index)
            gathered_query = torch.stack(gathered_query, dim=0)
            gathered_key = torch.stack(gathered_key, dim=0)
            gathered_index = torch.stack(gathered_index, dim=0)
            query = gathered_query.view(-1, embedding_dim)
            key = gathered_key.view(-1, embedding_dim)
            offset = torch.arange(0, world_size * num_samples, num_samples, device=index.device)
            gathered_index = gathered_index + offset.unsqueeze(dim=-1)
            index = gathered_index.view(-1)

        _validate_shapes(query, key, index)

        similarity = F.cosine_similarity(query.unsqueeze(dim=-2), key, dim=-1)
        num_samples = similarity.size(-1)
        similarity = similarity.view(-1, num_samples)
        index = index.view(-1)

        _, retrieved_indices = torch.topk(similarity, k=num_samples, dim=-1)

        for _retrieved_indices, _idx in zip(retrieved_indices, index):
            _retrieved_indices = _retrieved_indices.tolist()
            _idx = _idx.item()
            _retrieved_rank = _retrieved_indices.index(_idx)

            super().update(_retrieved_rank + mink)


def _validate_shapes(
    query: torch.Tensor,
    key: torch.Tensor,
    index: torch.LongTensor,
) -> None:
    # TODO: support key.dim() is 3
    assert query.dim() == 2
    assert key.dim() == 2
    assert index.dim() == 1
