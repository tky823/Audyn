from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "kmeans_clustering",
]


def kmeans_clustering(
    input: torch.Tensor,
    centroids: Optional[torch.Tensor] = None,
    n_iter: int = 1,
    num_clusters: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """K-means clustering.

    Args:
        input (torch.Tensor): Features of shape (batch_size, embedding_dim).
        centroids (torch.Tensor, optional): Clustering centroids of shape
            (num_clusters, embedding_dim). If not given, they are initialized with input.
        n_iter (int): Number of iterations. Default: ``1``.
        num_clusters (int, optional): Number of clusters. This parameter is required when
            ``centroids`` is not given.

    Returns:
        tuple: Tuple of tensors

            - torch.LongTensor: Cluster indices of each sample.
            - torch.Tensor: Updated centroids.

    """
    assert input.dim() == 2, "Only 2D input is supported."

    dtype = input.dtype
    device = input.device

    if centroids is None:
        # initialize centroids
        assert num_clusters is not None, "Set num_clusters."

        embedding_dim = input.size(-1)

        factory_kwargs = {
            "dtype": dtype,
            "device": device,
        }
        centroids = torch.empty(
            (num_clusters, embedding_dim),
            **factory_kwargs,
        )

        # TODO: support DDP
        indices = torch.randperm(input.size(0))[:num_clusters]
        indices = indices.tolist()
        centroids = input[indices]
        centroids.copy_(centroids)
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
        norm = torch.sum(centroids**2, dim=-1)
        dot = torch.matmul(input, centroids.transpose(1, 0))
        distance = norm - 2 * dot
        indices = torch.argmin(distance, dim=-1)
        assignments = F.one_hot(indices, num_classes=num_clusters)
        assignments = assignments.permute(1, 0)
        prod = torch.matmul(assignments.to(dtype), input)
        num_assignments = assignments.sum(dim=-1, keepdim=True)
        centroids = prod / num_assignments.to(dtype)

    return indices, centroids
