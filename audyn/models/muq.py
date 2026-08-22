from typing import Tuple

import torch
import torch.nn as nn
from packaging import version

from ..modules.muq import VectorQuantizer
from .vqvae import VQVAE as VQVAE

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class MuQRVQ(nn.Module):
    def __init__(
        self,
        in_channels: int,
        codebook_size: int,
        embedding_dim: int,
        num_stages: int,
        downsample_rate: int = 4,
    ) -> None:
        super().__init__()

        backbone = []

        for _ in range(num_stages):
            encoder = nn.Conv1d(
                in_channels * downsample_rate, embedding_dim, kernel_size=1, stride=1
            )
            decoder = nn.Conv1d(
                embedding_dim, in_channels * downsample_rate, kernel_size=1, stride=1
            )
            vector_quantizer = VectorQuantizer(codebook_size, embedding_dim)
            layer = VQVAE(encoder, decoder, vector_quantizer=vector_quantizer)
            backbone.append(layer)

        self.backbone = nn.ModuleList(backbone)

        self.in_channels = in_channels
        self.codebook_size = codebook_size
        self.num_stages = num_stages
        self.downsample_rate = downsample_rate

        self.registered_weight_norms = set()

        self.weight_norm_()

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Forward pass of MuQRVQ."""
        downsample_rate = self.downsample_rate

        batch_size, in_channels, num_frames = input.size()

        assert num_frames % downsample_rate == 0

        x = input.view(batch_size, in_channels, num_frames // downsample_rate, downsample_rate)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, downsample_rate * in_channels, num_frames // downsample_rate)

        reconstructed = 0
        encoded = []
        quantized = []
        residual = []
        indices = []

        for index, layer in enumerate(self.backbone):
            _output, _encoded, _quantized, _indices = layer(x)
            reconstructed = reconstructed + _output

            encoded.append(_encoded)
            quantized.append(_quantized)
            residual.append(x)
            indices.append(_indices)

            x = x - _output

        encoded = torch.stack(encoded, dim=1)
        quantized = torch.stack(quantized, dim=1)
        residual = torch.stack(residual, dim=1)
        indices = torch.stack(indices, dim=1)

        x = reconstructed.view(
            batch_size, downsample_rate, in_channels, num_frames // downsample_rate
        )
        x = x.permute(0, 2, 3, 1)
        output = x.reshape(batch_size, in_channels, num_frames)

        return output, encoded, quantized, residual, indices

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        if "backbone" not in self.registered_weight_norms:
            for layer in self.backbone:
                layer: VQVAE
                layer.encoder = weight_norm_fn(layer.encoder)
                layer.decoder = weight_norm_fn(layer.decoder)

            self.registered_weight_norms.add("backbone")
