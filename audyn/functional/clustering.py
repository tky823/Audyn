from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

__all__ = [
    "kmeans_clustering",
    "initialize_centroids",
]


def kmeans_clustering(
    input: torch.Tensor,
    centroids: Optional[torch.Tensor] = None,
    n_iter: int = 1,
    num_clusters: Optional[int] = None,
    seed: int = 0,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """K-means clustering.

    Args:
        input (torch.Tensor): Features of shape (batch_size, embedding_dim).
        centroids (torch.Tensor, optional): Clustering centroids of shape
            (num_clusters, embedding_dim). If not given, they are initialized with input.
        n_iter (int): Number of iterations. Default: ``1``.
        num_clusters (int, optional): Number of clusters. This parameter is required when
            ``centroids`` is not given.
        seed (int): Random seed to select initial centroids. Default: ``0``.

    Returns:
        tuple: Tuple of tensors

            - torch.LongTensor: Cluster indices of each sample.
            - torch.Tensor: Updated centroids.

    """
    assert input.dim() == 2, "Only 2D input is supported."

    dtype = input.dtype

    is_distributed = dist.is_available() and dist.is_initialized()

    if centroids is None:
        # initialize centroids
        assert num_clusters is not None, "Set num_clusters."

        centroids = initialize_centroids(input, num_clusters=num_clusters, seed=seed)
    else:
        if num_clusters is None:
            num_clusters = centroids.size(0)
        else:
            expected_num_clusters = centroids.size(0)

            assert num_clusters == expected_num_clusters, (
                f"num_clusters={num_clusters} is inconsistent "
                f"with centroids.size(0)={expected_num_clusters}."
            )

        assert centroids.size(-1) == input.size(
            -1
        ), "Feature dimension of centroids is different from input."

    for _ in range(n_iter):
        indices = _compute_nearest_centroid_indices(input, centroids=centroids)
        assignments = F.one_hot(indices, num_classes=num_clusters)
        assignments = assignments.permute(1, 0)
        prod = torch.matmul(assignments.to(dtype), input)
        num_assignments = assignments.sum(dim=-1, keepdim=True)

        if is_distributed:
            dist.all_reduce(prod)
            dist.all_reduce(num_assignments)

        centroids = prod / num_assignments.to(dtype)

    indices = _compute_nearest_centroid_indices(input, centroids=centroids)

    return indices, centroids


def initialize_centroids(input: torch.Tensor, num_clusters: int, seed: int = 0) -> torch.Tensor:
    """Initialize centroids by selecting samples at random.

    .. note::

        When DDP is enabled, inputs are gathered and centroids are synchronized among devices.

    """
    is_distributed = dist.is_available() and dist.is_initialized()

    # to share random state among devices
    g = torch.Generator(device=input.device)
    g.manual_seed(seed)

    if is_distributed:
        # gather input
        # (batch_size, embedding_dim) -> (num_gpus * batch_size, embedding_dim)
        gathered_input = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_input, input)
        gathered_input = torch.concat(gathered_input, dim=0)
    else:
        gathered_input = input

    indices = torch.randperm(
        gathered_input.size(0),
        generator=g,
        dtype=torch.long,
    )
    indices = indices[:num_clusters].tolist()
    centroids = gathered_input[indices]

    return centroids


def _compute_nearest_centroid_indices(
    input: torch.Tensor, centroids: torch.Tensor
) -> torch.LongTensor:
    norm = torch.sum(centroids**2, dim=-1)
    dot = torch.matmul(input, centroids.transpose(1, 0))
    distance = norm - 2 * dot
    indices = torch.argmin(distance, dim=-1)

    return indices
