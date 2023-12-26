import copy
import math
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from dummy import allclose
from torch.optim import SGD, Adam

from audyn.modules.vqvae import VectorQuantizer
from audyn.optim.optimizer import (
    ExponentialMovingAverageCodebookOptimizer,
    ExponentialMovingAverageWrapper,
    MultiOptimizers,
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


@pytest.mark.parametrize("codebook_reset", [True, False])
def test_exponential_moving_average_codebook_optimizer(codebook_reset: bool) -> None:
    torch.manual_seed(0)

    codebook_size, embedding_dim = 3, 4
    batch_size, length = 2, 5
    num_initial_steps, num_total_steps = 5, 10
    reset_step, reset_var = 3, 0.1

    with tempfile.TemporaryDirectory() as temp_dir:
        model = VectorQuantizer(codebook_size, embedding_dim)

        if codebook_reset:
            optimizer = ExponentialMovingAverageCodebookOptimizer(
                model.parameters(),
                reset_step=reset_step,
                reset_var=reset_var,
            )
        else:
            optimizer = ExponentialMovingAverageCodebookOptimizer(model.parameters())

        model.register_forward_hook(optimizer.store_current_stats)
        input = torch.randn((batch_size, embedding_dim, length))

        for _ in range(num_initial_steps):
            optimizer.zero_grad()
            _, _ = model(input)
            optimizer.step()

        state_dict_stop = {}
        state_dict_stop["model"] = model.state_dict()
        state_dict_stop["optimizer"] = optimizer.state_dict()

        path = os.path.join(temp_dir, "stop.pth")
        torch.save(state_dict_stop, path)

        for _ in range(num_initial_steps, num_total_steps):
            optimizer.zero_grad()
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
        state_dict_stop = torch.load(path)

        model.load_state_dict(state_dict_stop["model"])
        optimizer.load_state_dict(state_dict_stop["optimizer"])

        # resume training from checkpoint
        for _ in range(num_initial_steps, num_total_steps):
            optimizer.zero_grad()
            output, _ = model(input)
            loss = output.mean()
            loss.backward()
            optimizer.step()

        state_dict_resume = {}
        state_dict_resume["model"] = model.state_dict()
        state_dict_resume["optimizer"] = optimizer.state_dict()

        path = os.path.join(temp_dir, "sequential.pth")
        state_dict_sequential = torch.load(path)

        if codebook_reset:
            # When codebook_reset=True, optimizer uses torch.randn, which violates reproducibility.
            return

        for (k_sequential, v_sequential), (k_resume, v_resume) in zip(
            state_dict_sequential["model"].items(), state_dict_resume["model"].items()
        ):
            assert k_sequential == k_resume
            assert torch.allclose(v_sequential, v_resume)

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
