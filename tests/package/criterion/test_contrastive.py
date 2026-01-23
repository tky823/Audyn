import copy
import itertools
import os
import sys
import tempfile
from datetime import timedelta
from typing import Tuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from audyn_test import allclose
from audyn_test.utils import select_random_port
from audyn_test.utils.ddp import retry_on_file_not_found, set_ddp_environment
from torch.optim import SGD

from audyn.criterion.contrastive import (
    InfoNCELoss,
    InterInfoNCELoss,
    InterNTXentLoss,
    IntraInfoNCELoss,
    IntraNTXentLoss,
    NTXentLoss,
)

IS_LINUX = sys.platform.startswith("linux")

# Use forkserver on Unix-like systems (faster), spawn on Windows (only option)
_MP_START_METHOD = "forkserver" if sys.platform != "win32" else "spawn"


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_info_nce_loss(reduction: str) -> None:
    torch.manual_seed(0)

    # 3D input
    batch_size, length, embedding_dim = 4, 12, 10

    input = torch.randn((batch_size, length, embedding_dim))
    other = torch.randn((batch_size, length, embedding_dim))

    criterion = InfoNCELoss(dim=0, reduction=reduction)
    reference_criterion = InterInfoNCELoss(dim=0, reduction=reduction)
    loss = criterion(input, other)
    reference_loss = reference_criterion(input, other)

    if reduction in ["mean", "sum"]:
        assert loss.size() == ()
    else:
        assert loss.size() == (length, batch_size)

    allclose(loss, reference_loss)

    criterion = InfoNCELoss(dim=1, reduction=reduction)
    reference_criterion = IntraInfoNCELoss(dim=1, reduction=reduction)
    loss = criterion(input, other)
    reference_loss = reference_criterion(input, other)

    if reduction in ["mean", "sum"]:
        assert loss.size() == ()
    else:
        assert loss.size() == (batch_size, length)

    allclose(loss, reference_loss)

    # 4D input
    batch_size, height, width, embedding_dim = 4, 5, 6, 10

    input = torch.randn((batch_size, height, width, embedding_dim))
    other = torch.randn((batch_size, height, width, embedding_dim))

    criterion = InfoNCELoss(dim=0, reduction=reduction)
    reference_criterion = InterInfoNCELoss(dim=0, reduction=reduction)
    loss = criterion(input, other)
    reference_loss = reference_criterion(input, other)

    if reduction in ["mean", "sum"]:
        assert loss.size() == ()
    else:
        assert loss.size() == (height, width, batch_size)

    allclose(loss, reference_loss)

    criterion = InfoNCELoss(dim=1, reduction=reduction)
    reference_criterion = IntraInfoNCELoss(dim=1, reduction=reduction)
    loss = criterion(input, other)
    reference_loss = reference_criterion(input, other)

    if reduction in ["mean", "sum"]:
        assert loss.size() == ()
    else:
        assert loss.size() == (batch_size, width, height)

    allclose(loss, reference_loss)

    criterion = InfoNCELoss(dim=2, reduction=reduction)
    reference_criterion = IntraInfoNCELoss(dim=2, reduction=reduction)
    loss = criterion(input, other)
    reference_loss = reference_criterion(input, other)

    if reduction in ["mean", "sum"]:
        assert loss.size() == ()
    else:
        assert loss.size() == (batch_size, height, width)

    allclose(loss, reference_loss)


@retry_on_file_not_found(3)
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_info_nce_loss_ddp(dim: int) -> None:
    """Ensure InterInfoNCELoss and IntraInfoNCELoss work well for DDP."""
    port = select_random_port()
    seed = 0

    if IS_LINUX:
        world_size = 3
    else:
        world_size = 2

    batch_size = 4

    torch.manual_seed(seed)
    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = mp.get_context(_MP_START_METHOD)

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = ctx.Process(
                target=run_contrastive_loss,
                args=(rank, world_size, port),
                kwargs={
                    "batch_size": batch_size,
                    "dim": dim,
                    "criterion_cls": InfoNCELoss,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            _ = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}_intra-or-inter.pth")

            if dim == 0:
                criterion_cls = InterInfoNCELoss
            else:
                criterion_cls = IntraInfoNCELoss

            process = ctx.Process(
                target=run_contrastive_loss,
                args=(rank, world_size, port),
                kwargs={
                    "batch_size": batch_size,
                    "dim": dim,
                    "criterion_cls": criterion_cls,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            state_dict = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            loss = state_dict["loss"]

            reference_path = os.path.join(temp_dir, f"{rank}_intra-or-inter.pth")
            reference_state_dict = torch.load(
                reference_path,
                map_location="cpu",
                weights_only=True,
            )
            reference_loss = reference_state_dict["loss"]

            assert torch.equal(loss, reference_loss)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ntxent_loss(reduction: str) -> None:
    torch.manual_seed(0)

    # 3D input
    batch_size, length, embedding_dim = 4, 12, 10

    input = torch.randn((batch_size, length, embedding_dim))
    other = torch.randn((batch_size, length, embedding_dim))

    criterion = NTXentLoss(dim=0, reduction=reduction)
    reference_criterion = InterNTXentLoss(dim=0, reduction=reduction)
    loss = criterion(input, other)
    reference_loss = reference_criterion(input, other)

    if reduction in ["mean", "sum"]:
        assert loss.size() == ()
    else:
        assert loss.size() == (length, batch_size)

    allclose(loss, reference_loss)

    criterion = NTXentLoss(dim=1, reduction=reduction)
    reference_criterion = IntraNTXentLoss(dim=1, reduction=reduction)
    loss = criterion(input, other)
    reference_loss = reference_criterion(input, other)

    if reduction in ["mean", "sum"]:
        assert loss.size() == ()
    else:
        assert loss.size() == (batch_size, length)

    allclose(loss, reference_loss)

    # 4D input
    batch_size, height, width, embedding_dim = 4, 5, 6, 10

    input = torch.randn((batch_size, height, width, embedding_dim))
    other = torch.randn((batch_size, height, width, embedding_dim))

    criterion = NTXentLoss(dim=0, reduction=reduction)
    reference_criterion = InterNTXentLoss(dim=0, reduction=reduction)
    loss = criterion(input, other)
    reference_loss = reference_criterion(input, other)

    if reduction in ["mean", "sum"]:
        assert loss.size() == ()
    else:
        assert loss.size() == (height, width, batch_size)

    allclose(loss, reference_loss)

    criterion = NTXentLoss(dim=1, reduction=reduction)
    reference_criterion = IntraNTXentLoss(dim=1, reduction=reduction)
    loss = criterion(input, other)
    reference_loss = reference_criterion(input, other)

    if reduction in ["mean", "sum"]:
        assert loss.size() == ()
    else:
        assert loss.size() == (batch_size, width, height)

    allclose(loss, reference_loss)

    criterion = NTXentLoss(dim=2, reduction=reduction)
    reference_criterion = IntraNTXentLoss(dim=2, reduction=reduction)
    loss = criterion(input, other)
    reference_loss = reference_criterion(input, other)

    if reduction in ["mean", "sum"]:
        assert loss.size() == ()
    else:
        assert loss.size() == (batch_size, height, width)

    allclose(loss, reference_loss)


@retry_on_file_not_found(3)
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_ntxent_loss_ddp(dim: int) -> None:
    """Ensure InterNTXentLoss and IntraNTXentLoss work well for DDP."""
    port = select_random_port()
    seed = 0

    if IS_LINUX:
        world_size = 3
    else:
        world_size = 2

    batch_size = 2

    torch.manual_seed(seed)
    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = mp.get_context(_MP_START_METHOD)

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = ctx.Process(
                target=run_contrastive_loss,
                args=(rank, world_size, port),
                kwargs={
                    "batch_size": batch_size,
                    "dim": dim,
                    "criterion_cls": NTXentLoss,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            _ = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}_intra-or-inter.pth")

            if dim == 0:
                criterion_cls = InterNTXentLoss
            else:
                criterion_cls = IntraNTXentLoss

            process = ctx.Process(
                target=run_contrastive_loss,
                args=(rank, world_size, port),
                kwargs={
                    "batch_size": batch_size,
                    "dim": dim,
                    "criterion_cls": criterion_cls,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            state_dict = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            loss = state_dict["loss"]

            reference_path = os.path.join(temp_dir, f"{rank}_intra-or-inter.pth")
            reference_state_dict = torch.load(
                reference_path,
                map_location="cpu",
                weights_only=True,
            )
            reference_loss = reference_state_dict["loss"]

            assert torch.equal(loss, reference_loss)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_intra_info_nce_loss(reduction: str) -> None:
    torch.manual_seed(0)

    # 3D input
    batch_size, length, embedding_dim = 4, 12, 10

    input = torch.randn((batch_size, length, embedding_dim))
    other = torch.randn((batch_size, length, embedding_dim))

    criterion = IntraInfoNCELoss(dim=0, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (length, batch_size)
        assert loss2.size() == (length, batch_size)

    allclose(loss1, loss2)

    criterion = IntraInfoNCELoss(dim=1, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (batch_size, length)
        assert loss2.size() == (batch_size, length)

    allclose(loss1, loss2)

    # 4D input
    batch_size, height, width, embedding_dim = 4, 5, 6, 10

    input = torch.randn((batch_size, height, width, embedding_dim))
    other = torch.randn((batch_size, height, width, embedding_dim))

    criterion = IntraInfoNCELoss(dim=0, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (height, width, batch_size)
        assert loss2.size() == (height, width, batch_size)

    allclose(loss1, loss2)

    criterion = IntraInfoNCELoss(dim=1, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (batch_size, width, height)
        assert loss2.size() == (batch_size, width, height)

    allclose(loss1, loss2)

    criterion = IntraInfoNCELoss(dim=2, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (batch_size, height, width)
        assert loss2.size() == (batch_size, height, width)

    allclose(loss1, loss2)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_intra_ntxent_loss(reduction: str) -> None:
    torch.manual_seed(0)

    # 3D input
    batch_size, length, embedding_dim = 4, 12, 10

    input = torch.randn((batch_size, length, embedding_dim))
    other = torch.randn((batch_size, length, embedding_dim))

    criterion = IntraNTXentLoss(dim=0, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (length, batch_size)
        assert loss2.size() == (length, batch_size)

    allclose(loss1, loss2)

    criterion = IntraNTXentLoss(dim=1, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (batch_size, length)
        assert loss2.size() == (batch_size, length)

    allclose(loss1, loss2)

    # 4D input
    batch_size, height, width, embedding_dim = 4, 5, 6, 10

    input = torch.randn((batch_size, height, width, embedding_dim))
    other = torch.randn((batch_size, height, width, embedding_dim))

    criterion = IntraNTXentLoss(dim=0, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (height, width, batch_size)
        assert loss2.size() == (height, width, batch_size)

    allclose(loss1, loss2)

    criterion = IntraNTXentLoss(dim=1, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (batch_size, width, height)
        assert loss2.size() == (batch_size, width, height)

    allclose(loss1, loss2)

    criterion = IntraNTXentLoss(dim=2, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (batch_size, height, width)
        assert loss2.size() == (batch_size, height, width)

    allclose(loss1, loss2)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_inter_info_nce_loss(reduction: str) -> None:
    torch.manual_seed(0)

    # 3D input
    batch_size, length, embedding_dim = 4, 12, 10

    input = torch.randn((batch_size, length, embedding_dim))
    other = torch.randn((batch_size, length, embedding_dim))

    criterion = InterInfoNCELoss(dim=0, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (length, batch_size)
        assert loss2.size() == (length, batch_size)

    allclose(loss1, loss2)

    criterion = InterInfoNCELoss(dim=1, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (batch_size, length)
        assert loss2.size() == (batch_size, length)

    allclose(loss1, loss2)

    # 4D input
    batch_size, height, width, embedding_dim = 4, 5, 6, 10

    input = torch.randn((batch_size, height, width, embedding_dim))
    other = torch.randn((batch_size, height, width, embedding_dim))

    criterion = InterInfoNCELoss(dim=0, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (height, width, batch_size)
        assert loss2.size() == (height, width, batch_size)

    allclose(loss1, loss2)

    criterion = InterInfoNCELoss(dim=1, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (batch_size, width, height)
        assert loss2.size() == (batch_size, width, height)

    allclose(loss1, loss2)

    criterion = InterInfoNCELoss(dim=2, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (batch_size, height, width)
        assert loss2.size() == (batch_size, height, width)

    allclose(loss1, loss2)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_inter_ntxent_loss(reduction: str) -> None:
    torch.manual_seed(0)

    # 3D input
    batch_size, length, embedding_dim = 4, 12, 10

    input = torch.randn((batch_size, length, embedding_dim))
    other = torch.randn((batch_size, length, embedding_dim))

    criterion = InterNTXentLoss(dim=0, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (length, batch_size)
        assert loss2.size() == (length, batch_size)

    allclose(loss1, loss2)

    criterion = InterNTXentLoss(dim=1, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (batch_size, length)
        assert loss2.size() == (batch_size, length)

    allclose(loss1, loss2)

    # 4D input
    batch_size, height, width, embedding_dim = 4, 5, 6, 10

    input = torch.randn((batch_size, height, width, embedding_dim))
    other = torch.randn((batch_size, height, width, embedding_dim))

    criterion = InterNTXentLoss(dim=0, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (height, width, batch_size)
        assert loss2.size() == (height, width, batch_size)

    allclose(loss1, loss2)

    criterion = InterNTXentLoss(dim=1, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (batch_size, width, height)
        assert loss2.size() == (batch_size, width, height)

    allclose(loss1, loss2)

    criterion = InterNTXentLoss(dim=2, reduction=reduction)
    loss1 = criterion(input, other)
    loss2 = criterion(other, input)

    if reduction in ["mean", "sum"]:
        assert loss1.size() == ()
        assert loss2.size() == ()
    else:
        assert loss1.size() == (batch_size, height, width)
        assert loss2.size() == (batch_size, height, width)

    allclose(loss1, loss2)


@retry_on_file_not_found(3)
@pytest.mark.parametrize("dim", [0, 1])
def test_inter_info_nce_loss_ddp(dim: int) -> None:
    """Ensure InterInfoNCELoss works well for DDP."""
    port = select_random_port()
    seed = 0

    if IS_LINUX:
        world_size = 3
    else:
        world_size = 2

    batch_size = 2
    in_channels, out_channels = 2, 3
    lr = 0.1
    iterations = 10

    torch.manual_seed(seed)
    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = mp.get_context(_MP_START_METHOD)

        # multiple devices
        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = ctx.Process(
                target=run_inter_info_nce_loss,
                args=(rank, world_size, port),
                kwargs={
                    "batch_size": batch_size,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "dim": dim,
                    "lr": lr,
                    "iterations": iterations,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        gathered_input = []
        gathered_other = []

        rank = 0
        reference_modules = build_inter_info_nce_modules(
            in_channels, out_channels, dim=dim, is_distributed=False
        )
        reference_model_one, reference_model_other, reference_criterion = reference_modules

        path = os.path.join(temp_dir, f"{rank}.pth")
        reference_state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        ddp_state_dict = reference_state_dict["last"]
        reference_model_one.load_state_dict(ddp_state_dict["model_one"])
        reference_model_other.load_state_dict(ddp_state_dict["model_other"])
        reference_criterion.load_state_dict(ddp_state_dict["criterion"])

        input = reference_state_dict["input"]
        other = reference_state_dict["other"]
        gathered_input.append(input)
        gathered_other.append(other)

        for rank in range(1, world_size):
            modules = build_inter_info_nce_modules(
                in_channels, out_channels, dim=dim, is_distributed=False
            )
            model_one, model_other, criterion = modules

            path = os.path.join(temp_dir, f"{rank}.pth")
            state_dict = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            ddp_state_dict = state_dict["last"]
            model_one.load_state_dict(ddp_state_dict["model_one"])
            model_other.load_state_dict(ddp_state_dict["model_other"])
            criterion.load_state_dict(ddp_state_dict["criterion"])

            input = state_dict["input"]
            other = state_dict["other"]
            gathered_input.append(input)
            gathered_other.append(other)

            assert len(list(model_one.parameters())) == len(list(reference_model_one.parameters()))

            for param, param_reference in zip(
                model_one.parameters(), reference_model_one.parameters()
            ):
                assert param.size() == param_reference.size()
                assert torch.equal(param, param_reference)

            assert len(list(model_other.parameters())) == len(
                list(reference_model_other.parameters())
            )

            for param, param_reference in zip(
                model_other.parameters(), reference_model_other.parameters()
            ):
                assert param.size() == param_reference.size()
                assert torch.equal(param, param_reference)

            assert len(list(criterion.parameters())) == len(list(reference_criterion.parameters()))

            for param, param_reference in zip(
                criterion.parameters(), reference_criterion.parameters()
            ):
                assert param.size() == param_reference.size()
                assert torch.equal(param, param_reference)

        # single device
        ddp_state_dict = reference_state_dict["initial"]
        reference_model_one.load_state_dict(ddp_state_dict["model_one"])
        reference_model_other.load_state_dict(ddp_state_dict["model_other"])
        reference_criterion.load_state_dict(ddp_state_dict["criterion"])
        reference_optimizer = SGD(
            itertools.chain(
                reference_model_one.parameters(),
                reference_model_other.parameters(),
                reference_criterion.parameters(),
            ),
            lr=lr,
        )

        gathered_input = torch.cat(gathered_input, dim=dim)
        gathered_other = torch.cat(gathered_other, dim=dim)

        _ = update_inter_contrastive_modules(
            gathered_input,
            gathered_other,
            reference_model_one,
            reference_model_other,
            reference_criterion,
            reference_optimizer,
            iterations=iterations,
        )

        ddp_state_dict = reference_state_dict["last"]
        no_ddp_state_dict = {
            "model_one": reference_model_one.state_dict(),
            "model_other": reference_model_other.state_dict(),
            "criterion": reference_criterion.state_dict(),
        }

        for key in ddp_state_dict.keys():
            module_ddp = ddp_state_dict[key]
            module_no_ddp = no_ddp_state_dict[key]

            for _key in module_ddp.keys():
                allclose(module_ddp[_key], module_no_ddp[_key])


@retry_on_file_not_found(3)
@pytest.mark.parametrize("dim", [0, 1])
def test_inter_ntxent_loss_ddp(dim: int) -> None:
    """Ensure InterNTXentLoss works well for DDP."""
    port = select_random_port()
    seed = 0

    if IS_LINUX:
        world_size = 3
    else:
        world_size = 2

    batch_size = 2
    in_channels, out_channels = 2, 3
    lr = 0.1
    iterations = 10

    torch.manual_seed(seed)
    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = mp.get_context(_MP_START_METHOD)

        # multiple devices
        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = ctx.Process(
                target=run_inter_ntxent_loss,
                args=(rank, world_size, port),
                kwargs={
                    "batch_size": batch_size,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "dim": dim,
                    "lr": lr,
                    "iterations": iterations,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        gathered_input = []
        gathered_other = []

        rank = 0
        reference_modules = build_inter_ntxent_modules(
            in_channels, out_channels, dim=dim, is_distributed=False
        )
        reference_model_one, reference_model_other, reference_criterion = reference_modules

        path = os.path.join(temp_dir, f"{rank}.pth")
        reference_state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        ddp_state_dict = reference_state_dict["last"]
        reference_model_one.load_state_dict(ddp_state_dict["model_one"])
        reference_model_other.load_state_dict(ddp_state_dict["model_other"])
        reference_criterion.load_state_dict(ddp_state_dict["criterion"])

        input = reference_state_dict["input"]
        other = reference_state_dict["other"]
        gathered_input.append(input)
        gathered_other.append(other)

        for rank in range(1, world_size):
            modules = build_inter_ntxent_modules(
                in_channels, out_channels, dim=dim, is_distributed=False
            )
            model_one, model_other, criterion = modules

            path = os.path.join(temp_dir, f"{rank}.pth")
            state_dict = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            ddp_state_dict = state_dict["last"]
            model_one.load_state_dict(ddp_state_dict["model_one"])
            model_other.load_state_dict(ddp_state_dict["model_other"])
            criterion.load_state_dict(ddp_state_dict["criterion"])

            input = state_dict["input"]
            other = state_dict["other"]
            gathered_input.append(input)
            gathered_other.append(other)

            assert len(list(model_one.parameters())) == len(list(reference_model_one.parameters()))

            for param, param_reference in zip(
                model_one.parameters(), reference_model_one.parameters()
            ):
                assert param.size() == param_reference.size()
                assert torch.equal(param, param_reference)

            assert len(list(model_other.parameters())) == len(
                list(reference_model_other.parameters())
            )

            for param, param_reference in zip(
                model_other.parameters(), reference_model_other.parameters()
            ):
                assert param.size() == param_reference.size()
                assert torch.equal(param, param_reference)

            assert len(list(criterion.parameters())) == len(list(reference_criterion.parameters()))

            for param, param_reference in zip(
                criterion.parameters(), reference_criterion.parameters()
            ):
                assert param.size() == param_reference.size()
                assert torch.equal(param, param_reference)

        # single device
        ddp_state_dict = reference_state_dict["initial"]
        reference_model_one.load_state_dict(ddp_state_dict["model_one"])
        reference_model_other.load_state_dict(ddp_state_dict["model_other"])
        reference_criterion.load_state_dict(ddp_state_dict["criterion"])
        reference_optimizer = SGD(
            itertools.chain(
                reference_model_one.parameters(),
                reference_model_other.parameters(),
                reference_criterion.parameters(),
            ),
            lr=lr,
        )

        gathered_input = torch.cat(gathered_input, dim=dim)
        gathered_other = torch.cat(gathered_other, dim=dim)

        _ = update_inter_contrastive_modules(
            gathered_input,
            gathered_other,
            reference_model_one,
            reference_model_other,
            reference_criterion,
            reference_optimizer,
            iterations=iterations,
        )

        ddp_state_dict = reference_state_dict["last"]
        no_ddp_state_dict = {
            "model_one": reference_model_one.state_dict(),
            "model_other": reference_model_other.state_dict(),
            "criterion": reference_criterion.state_dict(),
        }

        for key in ddp_state_dict.keys():
            module_ddp = ddp_state_dict[key]
            module_no_ddp = no_ddp_state_dict[key]

            for _key in module_ddp.keys():
                allclose(module_ddp[_key], module_no_ddp[_key])


def run_contrastive_loss(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    dim: int = 0,
    criterion_cls: nn.Module = None,
    seed: int = 0,
    path: str = None,
) -> None:
    in_channels = 2
    height, width = 6, 5

    set_ddp_environment(rank, world_size, port)

    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=10),
    )
    torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(rank)

    criterion = criterion_cls(dim=dim)

    input = torch.randn((batch_size, height, width, in_channels), generator=g)
    other = torch.randn((batch_size, height, width, in_channels), generator=g)
    loss = criterion(input, other)

    state_dict = {
        "input": input,
        "other": other,
        "loss": loss,
    }
    torch.save(state_dict, path)

    dist.destroy_process_group()


def run_inter_info_nce_loss(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    in_channels: int = 2,
    out_channels: int = 3,
    dim: int = 0,
    lr: float = 0.1,
    iterations: int = 10,
    seed: int = 0,
    path: str = None,
) -> None:
    length = 5

    set_ddp_environment(rank, world_size, port)

    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=10),
    )
    torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(rank)

    model_one, model_other, criterion = build_inter_info_nce_modules(
        in_channels, out_channels, dim=dim, is_distributed=True
    )
    optimizer = SGD(
        itertools.chain(model_one.parameters(), model_other.parameters(), criterion.parameters()),
        lr=lr,
    )

    input = torch.randn((batch_size, length, in_channels), generator=g)
    other = torch.randn((batch_size, length, in_channels), generator=g)

    state_dict = {
        "input": input,
        "other": other,
        "initial": {
            "model_one": copy.deepcopy(model_one.module.state_dict()),
            "model_other": copy.deepcopy(model_other.module.state_dict()),
            "criterion": copy.deepcopy(criterion.module.state_dict()),
        },
    }

    loss = update_inter_contrastive_modules(
        input, other, model_one, model_other, criterion, optimizer, iterations=iterations
    )

    state_dict.update(
        {
            "loss": loss,
            "last": {
                "model_one": model_one.module.state_dict(),
                "model_other": model_other.module.state_dict(),
                "criterion": criterion.module.state_dict(),
            },
        }
    )
    torch.save(state_dict, path)

    dist.destroy_process_group()


def run_inter_ntxent_loss(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    in_channels: int = 2,
    out_channels: int = 3,
    dim: int = 0,
    lr: float = 0.1,
    iterations: int = 10,
    seed: int = 0,
    path: str = None,
) -> None:
    length = 5

    set_ddp_environment(rank, world_size, port)

    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=10),
    )
    torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(rank)

    model_one, model_other, criterion = build_inter_ntxent_modules(
        in_channels, out_channels, dim=dim, is_distributed=True
    )
    optimizer = SGD(
        itertools.chain(model_one.parameters(), model_other.parameters(), criterion.parameters()),
        lr=lr,
    )

    input = torch.randn((batch_size, length, in_channels), generator=g)
    other = torch.randn((batch_size, length, in_channels), generator=g)

    state_dict = {
        "input": input,
        "other": other,
        "initial": {
            "model_one": copy.deepcopy(model_one.module.state_dict()),
            "model_other": copy.deepcopy(model_other.module.state_dict()),
            "criterion": copy.deepcopy(criterion.module.state_dict()),
        },
    }

    loss = update_inter_contrastive_modules(
        input, other, model_one, model_other, criterion, optimizer, iterations=iterations
    )

    state_dict.update(
        {
            "loss": loss,
            "last": {
                "model_one": model_one.module.state_dict(),
                "model_other": model_other.module.state_dict(),
                "criterion": criterion.module.state_dict(),
            },
        }
    )
    torch.save(state_dict, path)

    dist.destroy_process_group()


def build_inter_info_nce_modules(
    in_channels: int, out_channels: int, dim: int = 0, is_distributed: bool = False
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    model_one = nn.Linear(in_channels, out_channels)
    model_other = nn.Linear(in_channels, out_channels)
    criterion = InterInfoNCELoss(dim=dim)

    if is_distributed:
        model_one = nn.parallel.DistributedDataParallel(model_one)
        model_other = nn.parallel.DistributedDataParallel(model_other)
        criterion = nn.parallel.DistributedDataParallel(criterion)

    return model_one, model_other, criterion


def build_inter_ntxent_modules(
    in_channels: int, out_channels: int, dim: int = 0, is_distributed: bool = False
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    model_one = nn.Linear(in_channels, out_channels)
    model_other = nn.Linear(in_channels, out_channels)
    criterion = InterNTXentLoss(dim=dim)

    if is_distributed:
        model_one = nn.parallel.DistributedDataParallel(model_one)
        model_other = nn.parallel.DistributedDataParallel(model_other)
        criterion = nn.parallel.DistributedDataParallel(criterion)

    return model_one, model_other, criterion


def update_inter_contrastive_modules(
    input: torch.Tensor,
    other: torch.Tensor,
    model_one: nn.Module,
    model_other: nn.Module,
    criterion: nn.Module,
    optimizer: SGD,
    iterations: int = 1,
) -> torch.Tensor:
    for _ in range(iterations):
        output_one = model_one(input)
        output_other = model_other(other)
        loss = criterion(output_one, output_other)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss
