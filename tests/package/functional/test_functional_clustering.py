import os
import sys
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dummy import allclose
from dummy.utils import select_random_port
from dummy.utils.ddp import set_ddp_environment
from omegaconf import OmegaConf

from audyn.functional.clustering import kmeans_clustering

IS_WINDOWS = sys.platform == "win32"


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


def test_kmeans_clustering_ddp() -> None:
    port = select_random_port()
    seed = 0
    world_size = 2

    torch.manual_seed(seed)

    batch_size, embedding_dim = 10, 4
    n_iter = 10

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        processes = []

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = mp.Process(
                target=run_kmeans_clustering,
                args=(rank, world_size, port),
                kwargs={
                    "seed": seed,
                    "path": path,
                    "batch_size": batch_size,
                    "embedding_dim": embedding_dim,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        input = []

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            data_rank = torch.load(path)
            input.append(data_rank["input"])

        input = torch.cat(input, dim=0)

        indices, centroids = kmeans_clustering(
            input,
            num_clusters=world_size,
            n_iter=n_iter,
            seed=seed,
        )
        indices = indices.view(world_size, batch_size)

        for rank, indices_rank in enumerate(indices):
            path = os.path.join(temp_dir, f"{rank}.pth")
            data_rank = torch.load(path)

            assert data_rank["indices"].size() == (batch_size,)
            assert torch.equal(data_rank["indices"], indices_rank)

            assert data_rank["centroids"].size() == (world_size, embedding_dim)
            allclose(data_rank["centroids"], centroids)


def run_kmeans_clustering(
    rank: int,
    world_size: int,
    port: int,
    seed: int = 0,
    path: str = None,
    batch_size: int = None,
    embedding_dim: int = None,
) -> None:
    set_ddp_environment(rank, world_size, port)

    if IS_WINDOWS:
        init_method = f"tcp://localhost:{port}"
    else:
        init_method = None

    config = {
        "seed": seed,
        "distributed": {
            "enable": True,
            "backend": "gloo",
            "init_method": init_method,
        },
        "cudnn": {
            "benchmark": None,
            "deterministic": None,
        },
        "amp": {
            "enable": False,
            "accelerator": "cpu",
        },
    }

    config = OmegaConf.create(config)

    dist.init_process_group(
        backend=config.distributed.backend,
        init_method=config.distributed.init_method,
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        timeout=timedelta(minutes=1),
    )
    torch.manual_seed(config.seed)

    num_clusters = world_size
    n_iter = 10

    g = torch.Generator()
    g.manual_seed(rank)

    input = torch.randn(
        (batch_size, embedding_dim),
        generator=g,
    )
    input = input + rank

    indices, centroids = kmeans_clustering(
        input,
        num_clusters=num_clusters,
        n_iter=n_iter,
        seed=seed,
    )
    data = {
        "input": input,
        "indices": indices,
        "centroids": centroids,
    }

    torch.save(data, path)

    dist.destroy_process_group()
