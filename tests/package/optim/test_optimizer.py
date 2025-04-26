import copy
import math
import os
import sys
import tempfile
from datetime import timedelta
from typing import Callable, Optional, Union

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from dummy import allclose
from dummy.utils import select_random_port
from dummy.utils.ddp import retry_on_file_not_found, set_ddp_environment
from omegaconf import OmegaConf
from torch.optim import SGD, Adam

from audyn.criterion.negative_sampling import DistanceBasedNegativeSamplingLoss
from audyn.functional.poincare import poincare_distance
from audyn.models.rvqvae import RVQVAE
from audyn.models.vqvae import VQVAE
from audyn.modules import PoincareEmbedding
from audyn.modules.rvq import ResidualVectorQuantizer
from audyn.modules.vq import VectorQuantizer
from audyn.optim.optimizer import (
    ExponentialMovingAverageCodebookOptimizer,
    ExponentialMovingAverageWrapper,
    MultiOptimizers,
    RiemannSGD,
)


@pytest.mark.parametrize("build_from_optim_class", [True, False])
def test_exponential_moving_average_wrapper(build_from_optim_class: bool):
    """Confirm moving average works correctly."""
    torch.manual_seed(0)

    batch_size = 2
    in_channels, out_channels = 3, 5
    smooth = 0.8
    iterations = 2  # update twice

    model = CustomModel(in_channels, out_channels)

    if build_from_optim_class:
        optimizer = Adam(model.parameters())
        optimizer_wrapper = ExponentialMovingAverageWrapper(optimizer, smooth=smooth)
    else:
        optimizer_wrapper = ExponentialMovingAverageWrapper.build_from_optim_class(
            model.parameters(), optimizer_class=Adam, smooth=smooth
        )

    linear_weight_moving_average = copy.deepcopy(model.linear.weight.data.detach())
    norm_weight_moving_average = copy.deepcopy(model.norm.weight.data.detach())

    for _ in range(iterations):
        input = torch.randn((batch_size, in_channels))
        output = model(input)
        loss = torch.mean(output)

        model.zero_grad()
        loss.backward()
        optimizer_wrapper.step()

        linear_weight = copy.deepcopy(model.linear.weight.data.detach())
        norm_weight = copy.deepcopy(model.norm.weight.data.detach())
        linear_weight_moving_average = (
            smooth * linear_weight_moving_average + (1 - smooth) * linear_weight
        )
        norm_weight_moving_average = (
            smooth * norm_weight_moving_average + (1 - smooth) * norm_weight
        )

    state_dict = {}

    # store state dict of optimizer
    state_dict["optimizer"] = copy.deepcopy(optimizer_wrapper.state_dict())

    # store state dict of model
    state_dict["model"] = copy.deepcopy(model.state_dict())
    optimizer_wrapper.set_moving_average_model()
    state_dict["moving_average_model"] = copy.deepcopy(model.state_dict())

    allclose(model.linear.weight.data, linear_weight_moving_average)
    allclose(model.norm.weight.data, norm_weight_moving_average)

    optimizer_wrapper.remove_moving_average_model()

    assert torch.equal(model.linear.weight.data, linear_weight)
    assert torch.equal(model.norm.weight.data, norm_weight)

    model.load_state_dict(state_dict["model"])
    model.load_state_dict(state_dict["moving_average_model"])

    optimizer_wrapper.load_state_dict(state_dict["optimizer"])


@pytest.mark.parametrize("is_rvq", [True, False])
@pytest.mark.parametrize("codebook_reset", [True, False])
@pytest.mark.parametrize("reset_strategy", ["ath", "rth", None])
@pytest.mark.parametrize("reset_source", ["mru", "batch"])
@pytest.mark.parametrize("reset_scope", ["least", "all", 1, None])
def test_exponential_moving_average_codebook_optimizer(
    is_rvq: bool,
    codebook_reset: bool,
    reset_strategy: Optional[str],
    reset_source: str,
    reset_scope: Optional[Union[str, int]],
) -> None:
    torch.manual_seed(0)

    num_stages = 6
    codebook_size, embedding_dim = 3, 4
    batch_size, length = 2, 5
    num_initial_steps, num_total_steps = 5, 10
    reset_step, reset_var = 3, 0.1

    with tempfile.TemporaryDirectory() as temp_dir:
        if is_rvq:
            model = ResidualVectorQuantizer(
                codebook_size,
                embedding_dim,
                num_stages=num_stages,
                dropout=False,
            )
        else:
            model = VectorQuantizer(codebook_size, embedding_dim)

        if codebook_reset:
            if reset_strategy == "ath":
                reset_ath = 2
                reset_rth = None
            elif reset_strategy == "rth":
                reset_ath = None
                reset_rth = 0.5
            else:
                reset_ath = None
                reset_rth = None

            optimizer = ExponentialMovingAverageCodebookOptimizer(
                model.parameters(),
                reset_step=reset_step,
                reset_var=reset_var,
                reset_ath=reset_ath,
                reset_rth=reset_rth,
                reset_source=reset_source,
                reset_scope=reset_scope,
            )
        else:
            optimizer = ExponentialMovingAverageCodebookOptimizer(model.parameters())

        model.register_forward_hook(optimizer.store_current_stats)
        input = torch.randn((batch_size, embedding_dim, length))

        for _ in range(num_initial_steps):
            optimizer.zero_grad()

            if is_rvq:
                _, _, _ = model(input)
            else:
                _, _ = model(input)

            optimizer.step()

        state_dict_stop = {}
        state_dict_stop["model"] = model.state_dict()
        state_dict_stop["optimizer"] = optimizer.state_dict()

        path = os.path.join(temp_dir, "stop.pth")
        torch.save(state_dict_stop, path)

        for _ in range(num_initial_steps, num_total_steps):
            optimizer.zero_grad()

            if is_rvq:
                output, _, _ = model(input)
            else:
                output, _ = model(input)

            loss = output.mean()
            loss.backward()
            optimizer.step()

        state_dict_sequential = {}
        state_dict_sequential["model"] = model.state_dict()
        state_dict_sequential["optimizer"] = optimizer.state_dict()

        path = os.path.join(temp_dir, "sequential.pth")
        torch.save(state_dict_sequential, path)

        path = os.path.join(temp_dir, "stop.pth")
        state_dict_stop = torch.load(
            path,
            weights_only=True,
        )

        model.load_state_dict(state_dict_stop["model"])
        optimizer.load_state_dict(state_dict_stop["optimizer"])

        # resume training from checkpoint
        for _ in range(num_initial_steps, num_total_steps):
            optimizer.zero_grad()

            if is_rvq:
                output, _, _ = model(input)
            else:
                output, _ = model(input)

            loss = output.mean()
            loss.backward()
            optimizer.step()

        state_dict_resume = {}
        state_dict_resume["model"] = model.state_dict()
        state_dict_resume["optimizer"] = optimizer.state_dict()

        path = os.path.join(temp_dir, "sequential.pth")
        state_dict_sequential = torch.load(
            path,
            weights_only=True,
        )

        if codebook_reset:
            # When codebook_reset=True, optimizer uses torch.randn, which violates reproducibility.
            return

        for (k_sequential, v_sequential), (k_resume, v_resume) in zip(
            state_dict_sequential["model"].items(), state_dict_resume["model"].items()
        ):
            assert k_sequential == k_resume

            if isinstance(v_sequential, torch.Tensor):
                assert isinstance(v_resume, torch.Tensor)
                assert torch.allclose(v_sequential, v_resume)
            else:
                # is_initialized
                assert type(v_sequential) is bool and type(v_resume) is bool
                assert v_sequential == v_resume

        for (k_sequential, v_sequential), (k_resume, v_resume) in zip(
            state_dict_sequential["optimizer"].items(), state_dict_resume["optimizer"].items()
        ):
            assert k_sequential == k_resume

            if k_sequential.endswith("_state"):
                for (_k_sequential, _v_sequential), (_k_resume, _v_resume) in zip(
                    v_sequential.items(), v_resume.items()
                ):
                    assert _k_sequential == _k_resume

                    if isinstance(_v_sequential, torch.Tensor):
                        assert isinstance(_v_resume, torch.Tensor)
                        assert torch.allclose(_v_sequential, _v_resume)
                    else:
                        assert math.isclose(_v_sequential, _v_resume)
            elif k_sequential.endswith("_groups"):
                if k_sequential == "param_groups":
                    # dictionary
                    for _v_sequential, _v_resume in zip(v_sequential, v_resume):
                        for (__k_sequential, __v_sequential), (__k_resume, __v_resume) in zip(
                            _v_sequential.items(), _v_resume.items()
                        ):
                            assert __k_sequential == __k_resume
                            assert __v_sequential == __v_resume
                else:
                    # list
                    assert v_sequential == v_resume
            elif k_sequential in ["smooth", "seed", "iteration"]:
                assert v_sequential == v_resume
            else:
                raise ValueError(f"Invalid key {k_sequential} is found.")


@retry_on_file_not_found(3)
@pytest.mark.parametrize("is_rvq", [True, False])
def test_exponential_moving_average_codebook_optimizer_ddp(is_rvq: bool) -> None:
    """Ensure ExponentialMovingAverageCodebookOptimizer works well for DDP."""
    port = select_random_port()
    world_size = 2
    seed, another_seed = 0, 1

    in_channels = 2
    kernel_size, stride = 5, 4
    codebook_size = 5
    embedding_dim = 8

    if is_rvq:
        build_model = build_dummy_rvqvae
    else:
        build_model = build_dummy_vqvae

    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = mp.Process(
                target=train_exponential_moving_average_codebook_optimizer,
                args=(rank, world_size, port, build_model),
                kwargs={
                    "in_channels": in_channels,
                    "kernel_size": kernel_size,
                    "stride": stride,
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
        reference_model = build_model(
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            seed=another_seed,
        )
        path = os.path.join(temp_dir, f"{rank}.pth")
        state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        reference_model.load_state_dict(state_dict)

        for rank in range(1, world_size):
            model = build_model(
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                codebook_size=codebook_size,
                embedding_dim=embedding_dim,
                seed=another_seed,
            )
            path = os.path.join(temp_dir, f"{rank}.pth")
            state_dict = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict)

            assert len(list(model.parameters())) == len(list(reference_model.parameters()))

            for param, param_reference in zip(model.parameters(), reference_model.parameters()):
                assert param.size() == param_reference.size()
                assert torch.equal(param, param_reference)


def test_rvq_optimizer_correctness() -> None:
    """Ensure computation of residual features in optimizer."""
    torch.manual_seed(0)

    num_stages = 2
    codebook_size, embedding_dim = 3, 4
    batch_size, length = 2, 5

    model = ResidualVectorQuantizer(
        codebook_size,
        embedding_dim,
        num_stages=num_stages,
        dropout=False,
    )

    optimizer = ExponentialMovingAverageCodebookOptimizer(model.parameters())
    model.register_forward_hook(optimizer.store_current_stats)
    input = torch.randn((batch_size, embedding_dim, length))

    optimizer.zero_grad()
    _, residual_by_model, _ = model(input)
    residual_by_model = residual_by_model.permute(1, 0, 3, 2).contiguous()
    residual_by_model = residual_by_model.view(num_stages, -1, embedding_dim)

    assert len(optimizer.residual_groups) == 1

    for residual_group in optimizer.residual_groups:
        assert torch.allclose(residual_group, residual_by_model)


def test_riemann_sgd() -> None:
    num_embedings = 10
    embedding_dim = 2
    num_neg_samples = 3

    manifold = PoincareEmbedding(num_embedings, embedding_dim)
    criterion = DistanceBasedNegativeSamplingLoss(
        poincare_distance,
        positive_distance_kwargs={
            "curvature": manifold.curvature,
            "dim": -1,
        },
        negative_distance_kwargs={
            "curvature": manifold.curvature,
            "dim": -1,
        },
    )
    optimizer = RiemannSGD(
        manifold.parameters(),
        expmap=manifold.expmap,
        proj=manifold.proj,
    )

    anchor = torch.randint(0, num_embedings, (), dtype=torch.long)
    positive = torch.randint(0, num_embedings, (), dtype=torch.long)
    negative = torch.randint(0, num_embedings, (num_neg_samples,), dtype=torch.long)

    anchor = manifold(anchor)
    positive = manifold(positive)
    negative = manifold(negative)
    loss = criterion(anchor, positive, negative)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


@pytest.mark.parametrize(
    "optim_type",
    [
        "list_dict",
        "list_optim",
    ],
)
def test_multi_optimizers(optim_type: str) -> None:
    """Confirm moving average works correctly."""
    torch.manual_seed(0)

    batch_size = 2
    in_channels, out_channels = 3, 5
    iterations = 2  # update twice

    model = CustomModel(in_channels, out_channels)
    sgd_optimizer = SGD(model.linear.parameters(), lr=0.1)
    adam_optimizer = Adam(model.norm.parameters(), lr=0.01)

    if optim_type == "list_dict":
        optimizers = [
            {
                "name": "sgd",
                "optimizer": sgd_optimizer,
            },
            {
                "name": "adam",
                "optimizer": adam_optimizer,
            },
        ]
    elif optim_type == "list_optim":
        optimizers = [
            sgd_optimizer,
            adam_optimizer,
        ]
    else:
        raise ValueError(f"{type(optim_type)} is not suppored as optim_type.")

    optimizer = MultiOptimizers(optimizers)

    for _ in range(iterations):
        input = torch.randn((batch_size, in_channels))
        output = model(input)
        loss = torch.mean(output)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()
        optimizer.load_state_dict(state_dict)


class CustomModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input (torch.Tensor): Tensor of shape (*, in_channels).

        Returns:
            torch.Tensor: Tensor of shape (*, out_channels).

        """
        x = self.linear(input)
        output = self.norm(x)

        return output


def train_exponential_moving_average_codebook_optimizer(
    rank: int,
    world_size: int,
    port: int,
    build_model: Callable,
    in_channels: int,
    kernel_size: int,
    stride: int = 1,
    codebook_size: int = 5,
    embedding_dim: int = 8,
    seed: int = 0,
    path: str = None,
) -> None:
    IS_WINDOWS = sys.platform == "win32"

    batch_size = 4
    height, width = 17, 17
    iterations = 5

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

    model = build_model(
        in_channels,
        kernel_size=kernel_size,
        stride=stride,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        seed=config.seed,
    )
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    unwrapped_model: VQVAE | RVQVAE = model.module
    optimizer = ExponentialMovingAverageCodebookOptimizer(
        unwrapped_model.vector_quantizer.parameters(),
        reset_step=1,
        reset_rate=0.9,
    )
    unwrapped_model.vector_quantizer.register_forward_hook(optimizer.store_current_stats)

    for _ in range(iterations):
        input = torch.randn((batch_size, in_channels, height, width), generator=g)

        if isinstance(unwrapped_model, RVQVAE):
            output, encoded, quantized, residual, indices = model(input)
        else:
            output, encoded, quantized, indices = model(input)
            residual = encoded

        assert output.size() == input.size()
        assert indices.size(0) == batch_size

        reconstrction_loss = torch.mean((output - input) ** 2)
        commitment_loss = torch.mean((residual - quantized.detach()) ** 2)
        codebook_loss = torch.mean((residual.detach() - quantized) ** 2)
        loss = reconstrction_loss + commitment_loss + codebook_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if isinstance(unwrapped_model, RVQVAE):
            quantized = quantized.transpose(1, 0)
            residual = residual.transpose(1, 0)
            num_stages = quantized.size(0)

            for stage_idx in range(num_stages):
                _residual = residual[stage_idx]

                if stage_idx == 0:
                    allclose(_residual, encoded)
                else:
                    _quantized = torch.sum(quantized[:stage_idx], dim=0)

                    allclose(_quantized + _residual, encoded, atol=1e-7)

    torch.save(model.module.state_dict(), path)

    dist.destroy_process_group()


def build_dummy_vqvae(
    in_channels: int,
    kernel_size: int,
    stride: int = 1,
    codebook_size: int = 5,
    embedding_dim: int = 8,
    seed: int = 0,
) -> nn.Module:
    model = _build_vqvae_or_rvqvae(
        in_channels,
        kernel_size=kernel_size,
        stride=stride,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        seed=seed,
        is_rvq=False,
    )
    return model


def build_dummy_rvqvae(
    in_channels: int,
    kernel_size: int,
    stride: int = 1,
    codebook_size: int = 5,
    embedding_dim: int = 8,
    seed: int = 0,
) -> nn.Module:
    model = _build_vqvae_or_rvqvae(
        in_channels,
        kernel_size=kernel_size,
        stride=stride,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        seed=seed,
        is_rvq=True,
    )

    return model


def _build_vqvae_or_rvqvae(
    in_channels: int,
    kernel_size: int,
    stride: int = 1,
    codebook_size: int = 5,
    embedding_dim: int = 8,
    seed: int = 0,
    is_rvq: bool = False,
) -> Union[VQVAE, RVQVAE]:
    padding = (kernel_size - 1) // 2

    encoder = nn.Conv2d(
        in_channels,
        embedding_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    decoder = nn.ConvTranspose2d(
        embedding_dim,
        in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    if is_rvq:
        num_stages = 3
        model = RVQVAE(
            encoder,
            decoder,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            num_stages=num_stages,
            seed=seed,
        )
    else:
        model = VQVAE(
            encoder,
            decoder,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            seed=seed,
        )

    return model
