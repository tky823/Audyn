import copy

import pytest
import torch
import torch.nn as nn
from dummy import allclose
from torch.optim import Adam

from audyn.optim.optimizer import ExponentialMovingAverageWrapper


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
