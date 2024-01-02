import copy
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import OmegaConf

from audyn.modules.rvq import ResidualVectorQuantizer


def test_residual_vector_quantizer() -> None:
    torch.manual_seed(0)

    batch_size = 4
    num_stages = 6
    codebook_size, embedding_dim = 10, 5
    length = 3

    input = torch.randn((batch_size, embedding_dim, length))

    rvq = ResidualVectorQuantizer(
        codebook_size,
        embedding_dim,
        num_stages=num_stages,
        dropout=False,
    )
    quantized, indices = rvq(input)

    assert quantized.size() == (batch_size, num_stages, embedding_dim, length)
    assert indices.size() == (batch_size, num_stages, length)

    # k-means clustering initalization
    kmeans_iteration = 100

    vector_quantizer = ResidualVectorQuantizer(
        codebook_size,
        embedding_dim,
        num_stages=num_stages,
        dropout=False,
        init_by_kmeans=kmeans_iteration,
    )

    _ = vector_quantizer(input)
    _, indices_before_save = vector_quantizer(input)
    state_dict = copy.copy(vector_quantizer.state_dict())
    vector_quantizer.load_state_dict(state_dict)

    _, indices_after_save = vector_quantizer(input)

    assert torch.equal(indices_before_save, indices_after_save)


def test_residual_vector_quantizer_ddp() -> None:
    """Ensure ResidualVectorQuantizer works well for DDP."""
    torch.manual_seed(0)

    port = str(torch.randint(0, 2**16, ()).item())
    world_size = 4
    seed, another_seed = 0, 1

    codebook_size = 5
    embedding_dim = 8

    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = mp.Process(
                target=train_dummy_rvqvae,
                args=(rank, world_size, port),
                kwargs={
                    "codebook_size": codebook_size,
                    "embedding_dim": embedding_dim,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        rank = 0
        reference_model = build_dummy_rvq(
            codebook_size,
            embedding_dim,
            seed=another_seed,
        )
        path = os.path.join(temp_dir, f"{rank}.pth")
        state_dict = torch.load(path, map_location="cpu")
        reference_model.load_state_dict(state_dict)

        for rank in range(1, world_size):
            model = build_dummy_rvq(
                codebook_size,
                embedding_dim,
                seed=seed,
            )
            path = os.path.join(temp_dir, f"{rank}.pth")
            state_dict = torch.load(path, map_location="cpu")
            model.load_state_dict(state_dict)

            assert len(list(model.parameters())) == len(list(reference_model.parameters()))

            for param, param_reference in zip(model.parameters(), reference_model.parameters()):
                assert param.size() == param_reference.size()
                assert torch.equal(param, param_reference)


def train_dummy_rvqvae(
    rank: int,
    world_size: int,
    port: int,
    codebook_size: int = 5,
    embedding_dim: int = 8,
    seed: int = 0,
    path: str = None,
) -> None:
    batch_size = 4
    length = 3

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    config = {
        "seed": seed,
        "distributed": {
            "enable": True,
            "backend": "gloo",
            "init_method": None,
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

    dist.init_process_group(backend=config.distributed.backend)
    torch.manual_seed(config.seed)

    g = torch.Generator()
    g.manual_seed(rank)

    model = build_dummy_rvq(codebook_size, embedding_dim, seed=seed)
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    input = torch.randn((batch_size, embedding_dim, length), generator=g)
    _ = model(input)

    torch.save(model.module.state_dict(), path)

    dist.destroy_process_group()


def build_dummy_rvq(
    codebook_size: int = 5,
    embedding_dim: int = 8,
    seed: int = 0,
) -> ResidualVectorQuantizer:
    num_stages = 6
    kmeans_iteration = 10

    model = ResidualVectorQuantizer(
        codebook_size,
        embedding_dim,
        num_stages=num_stages,
        dropout=True,
        init_by_kmeans=kmeans_iteration,
        seed=seed,
    )

    return model
