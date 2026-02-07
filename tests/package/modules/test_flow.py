from typing import Tuple

import torch
import torch.nn as nn
from audyn_test import allclose

from audyn.modules.flow import AdditiveCoupling, AffineCoupling, ChannelSplitFlow


class ReLUMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.linear = nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.linear(input)
        output = self.relu(x)

        return output


class AffineReLUMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.net1 = nn.Sequential(
            nn.Linear(in_features, out_features, bias=bias, **factory_kwargs), nn.ReLU()
        )
        self.net2 = nn.Sequential(
            nn.Linear(in_features, out_features, bias=bias, **factory_kwargs), nn.ReLU()
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_s = self.net1(input)
        t = self.net2(input)

        return log_s, t


def test_additive_coupling():
    torch.manual_seed(0)

    batch_size = 8
    split_channels = [2, 4]

    module = AdditiveCoupling(split_channels, coupling=ReLUMLP(*split_channels))
    input = torch.randn(batch_size, sum(split_channels))
    zeros = torch.zeros(batch_size)

    z = module(input)
    output = module(z, reverse=True)

    allclose(output, input)

    z, z_logdet = module(input, logdet=zeros)
    output, logdet = module(z, logdet=z_logdet, reverse=True)

    allclose(output, input)
    allclose(logdet, zeros)


def test_affine_coupling():
    torch.manual_seed(0)

    batch_size = 8
    split_channels = [2, 4]

    module = AffineCoupling(
        coupling=AffineReLUMLP(*split_channels),
        split=ChannelSplitFlow(split_channels),
    )
    input = torch.randn(batch_size, sum(split_channels))
    zeros = torch.zeros(batch_size)

    z = module(input)
    output = module(z, reverse=True)

    allclose(output, input)

    z, z_logdet = module(input, logdet=zeros)
    output, logdet = module(z, logdet=z_logdet, reverse=True)

    allclose(output, input)
    allclose(logdet, zeros)
