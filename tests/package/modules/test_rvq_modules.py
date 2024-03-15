import copy
import os
import sys
import tempfile
from datetime import timedelta
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from dummy.utils import set_ddp_environment
from omegaconf import OmegaConf
from torch.cuda.amp import autocast

from audyn.functional.vector_quantization import quantize_vector
from audyn.modules.rvq import ResidualVectorQuantizer

IS_WINDOWS = sys.platform == "win32"


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

    with tempfile.TemporaryDirectory() as temp_dir:
        initialization_path = os.path.join(temp_dir, "initialization.pth")
        forward_path = os.path.join(temp_dir, "forward.pth")

        vector_quantizer = CustomResidualVectorQuantizer(
            codebook_size,
            embedding_dim,
            num_stages=num_stages,
            dropout=False,
            init_by_kmeans=kmeans_iteration,
            initialization_path=initialization_path,
            forward_path=forward_path,
        )

        _ = vector_quantizer(input)

        initialization_output = torch.load(initialization_path, map_location="cpu")
        forward_output = torch.load(forward_path, map_location="cpu")
        forward_output = forward_output.permute(0, 3, 1, 2).contiguous()
        forward_output = forward_output.view(-1, num_stages, embedding_dim)

        assert torch.allclose(initialization_output, forward_output)


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
        path = os.path.join(temp_dir, "{rank}.pth")

        if IS_WINDOWS:
            mp.spawn(
                train_dummy_rvqvae,
                args=(
                    world_size,
                    port,
                    codebook_size,
                    embedding_dim,
                    seed,
                    path,
                ),
                nprocs=world_size,
            )
        else:
            for rank in range(world_size):
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


class CustomResidualVectorQuantizer(ResidualVectorQuantizer):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        num_stages: int,
        dropout: bool = True,
        init_by_kmeans: int = 0,
        seed: int = 0,
        initialization_path: str = None,
        forward_path: str = None,
    ) -> None:
        super().__init__(
            codebook_size,
            embedding_dim,
            num_stages=num_stages,
            dropout=dropout,
            init_by_kmeans=init_by_kmeans,
            seed=seed,
        )

        self.initialization_path = initialization_path
        self.forward_path = forward_path

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        quantized, indices = super().forward(input)

        torch.save(quantized, self.forward_path)

        return quantized, indices

    @torch.no_grad()
    def _initialize_parameters(self, encoded: torch.Tensor) -> None:
        assert self.init_by_kmeans > 0 and not self.is_initialized

        g = torch.Generator(device=encoded.device)
        g.manual_seed(self.seed)

        batch_size, embedding_dim, *_ = encoded.size()
        encoded = encoded.view(batch_size, embedding_dim, -1)
        encoded = encoded.permute(0, 2, 1).contiguous()
        encoded = encoded.view(-1, embedding_dim)
        stacked_quantized = []
        num_grids = encoded.size(0)
        reconstructed = 0

        with autocast(enabled=False):
            for codebook in self.codebooks:
                # select ``codebook_size`` embeddings from encoded features
                codebook: nn.Embedding
                codebook_size = codebook.weight.size(0)

                if num_grids < codebook_size:
                    msg = (
                        "Since number of grids given to RVQ is smaller than "
                        f"codebook size {codebook_size}, "
                        "we cannot apply k-means clustering initialization."
                    )
                    msg += " Please use larger batch size or smaller codebook size."

                    raise RuntimeError(msg)

                residual = encoded - reconstructed
                indices = torch.randperm(
                    num_grids,
                    generator=g,
                    device=residual.device,
                    dtype=torch.long,
                )
                indices = indices[:codebook_size]
                centroids = residual[indices]

                for _ in range(self.init_by_kmeans):
                    centroids = self._update_kmeans_centroids(residual, centroids)

                codebook.weight.data.copy_(centroids)
                quantized, _ = quantize_vector(residual, codebook.weight)
                reconstructed = reconstructed + quantized
                stacked_quantized.append(quantized)

        stacked_quantized = torch.stack(stacked_quantized, dim=1)
        torch.save(stacked_quantized, self.initialization_path)


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
    path = path.format(rank=rank)

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
        timeout=timedelta(minutes=5),
    )
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
