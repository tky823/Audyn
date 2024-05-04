import torch

from audyn.functional.clustering import kmeans_clustering


def test_kmeans_clustering() -> None:
    torch.manual_seed(0)

    batch_size_per_cluster, embedding_dim = 10, 2
    num_clusters = 3
    n_iter = 5

    # w/ initialized centroids
    input1 = 0.5 * torch.randn((batch_size_per_cluster, embedding_dim)) - 1
    input2 = torch.randn((batch_size_per_cluster, embedding_dim))
    input3 = torch.randn((batch_size_per_cluster, embedding_dim)) + 2
    input = torch.cat([input1, input2, input3], dim=0)
    indices = torch.randperm(input.size(0))[:num_clusters]
    indices = indices.tolist()
    centroids = input[indices]

    indices, centroids = kmeans_clustering(
        input,
        centroids=centroids,
        n_iter=n_iter,
    )

    # w/o initialized centroids
    input1 = 0.5 * torch.randn((batch_size_per_cluster, embedding_dim)) - 1
    input2 = torch.randn((batch_size_per_cluster, embedding_dim))
    input3 = torch.randn((batch_size_per_cluster, embedding_dim)) + 2
    input = torch.cat([input1, input2, input3], dim=0)

    indices, centroids = kmeans_clustering(
        input,
        num_clusters=num_clusters,
        n_iter=n_iter,
    )
