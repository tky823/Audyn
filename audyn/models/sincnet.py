from typing import List, Union

import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t

from ..modules.sincnet import ConvBlock1d, QuadratureSincConvBlock1d


class SincNet(nn.Module):
    def __init__(
        self,
        num_features: List[int],
        kernel_size: Union[List[_size_1_t], _size_1_t],
        pool_size: Union[List[_size_1_t], _size_1_t],
        sample_rate: int = 16000,
        stride: _size_1_t = 10,
    ) -> None:
        super().__init__()

        num_layers = len(num_features) - 1

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * num_layers

        if not isinstance(pool_size, list):
            pool_size = [pool_size] * num_layers

        assert len(kernel_size) == num_layers
        assert len(pool_size) == num_layers

        self.norm1d = nn.InstanceNorm1d(num_features[0], affine=True)

        backbone = []

        for layer_index in range(num_layers):
            in_channels = num_features[layer_index]
            out_channels = num_features[layer_index + 1]
            _kernel_size = kernel_size[layer_index]
            _pool_size = pool_size[layer_index]

            if layer_index == 0:
                block = QuadratureSincConvBlock1d(
                    in_channels,
                    out_channels,
                    kernel_size=_kernel_size,
                    pool_size=_pool_size,
                    stride=stride,
                    sample_rate=sample_rate,
                )
            else:
                block = ConvBlock1d(
                    in_channels,
                    out_channels,
                    kernel_size=_kernel_size,
                    pool_size=_pool_size,
                    stride=1,
                )

            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.norm1d(input)

        for block in self.backbone:
            x = block(x)

        output = x

        return output
