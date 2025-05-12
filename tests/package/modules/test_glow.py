import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from audyn_test import allclose
from omegaconf import OmegaConf

from audyn.modules.glow import (
    ActNorm1d,
    InvertiblePointwiseConv1d,
    InvertiblePointwiseConv2d,
)


def test_invertible_pointwise_conv1d():
    torch.manual_seed(0)

    batch_size = 2
    num_features = 6
    length = 16

    model = InvertiblePointwiseConv1d(num_features)
    input = torch.randn(batch_size, num_features, length)

    z = model(input)
    output = model(z, reverse=True)

    assert output.size() == input.size()
    assert z.size() == input.size()
    allclose(output, input, atol=1e-6)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input, atol=1e-6)
    allclose(logdet, zeros, atol=1e-7)


def test_invertible_pointwise_conv2d():
    torch.manual_seed(0)

    batch_size = 2
    num_features = 4
    height, width = 6, 6

    model = InvertiblePointwiseConv2d(num_features)
    input = torch.randn(batch_size, num_features, height, width)

    z = model(input)
    output = model(z, reverse=True)

    assert output.size() == input.size()
    assert z.size() == input.size()
    e = torch.abs(output - input)
    allclose(output, input, atol=1e-7), torch.max(e)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input, atol=1e-7)
    allclose(logdet, zeros, atol=1e-7)


def test_act_norm1d() -> None:
    torch.manual_seed(0)

    batch_size = 2
    num_features = 6
    length = 16

    model = ActNorm1d(num_features)
    input = torch.randn(batch_size, num_features, length)

    z = model(input)
    output = model(z, reverse=True)
    std, mean = torch.std_mean(z, dim=(0, 2), unbiased=False)

    assert output.size() == input.size()
    assert z.size() == input.size()
    allclose(output, input)
    allclose(mean, torch.zeros(()), atol=1e-7)
    allclose(std, torch.ones(()), atol=1e-7)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input)
    allclose(logdet, zeros)


def test_act_norm1d_ddp() -> None:
    """Ensure ActNorm1d works well for DDP."""
    torch.manual_seed(0)

    port = str(torch.randint(0, 2**16, ()).item())
    world_size = 2
    seed = 0

    num_features = 6

    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = mp.Process(
                target=train_dummy_act_norm1d,
                args=(rank, world_size, port),
                kwargs={
                    "num_features": num_features,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        z = []

        rank = 0
        reference_model = ActNorm1d(num_features)
        path = os.path.join(temp_dir, f"{rank}.pth")
        state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        reference_model.load_state_dict(state_dict["model"])
        z.append(state_dict["latent"])

        for rank in range(1, world_size):
            model = ActNorm1d(num_features)
            path = os.path.join(temp_dir, f"{rank}.pth")
            state_dict = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict["model"])
            z.append(state_dict["latent"])

            assert len(list(model.parameters())) == len(list(reference_model.parameters()))

            for param, param_reference in zip(model.parameters(), reference_model.parameters()):
                assert param.size() == param_reference.size()
                assert torch.equal(param, param_reference)

        z = torch.cat(z, dim=0)
        std, mean = torch.std_mean(z, dim=(0, 2), unbiased=False)

        allclose(mean, torch.zeros(()), atol=1e-7)
        allclose(std, torch.ones(()), atol=1e-7)


def train_dummy_act_norm1d(
    rank: int,
    world_size: int,
    port: int,
    num_features: int = 6,
    seed: int = 0,
    path: str = None,
) -> None:
    batch_size = 4
    length = 20

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    num_threads = torch.get_num_threads()
    num_threads = max(num_threads // world_size, 1)
    torch.set_num_threads(num_threads)

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

    model = ActNorm1d(num_features)
    model = nn.parallel.DistributedDataParallel(model)

    input = torch.randn((batch_size, num_features, length), generator=g)
    z = model(input)
    output = model(z, reverse=True)

    allclose(output, input)

    state_dict = {
        "latent": z,
        "model": model.module.state_dict(),
    }
    torch.save(state_dict, path)

    dist.destroy_process_group()
