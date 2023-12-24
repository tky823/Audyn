from typing import Tuple

import torch
import torch.nn as nn

from ..functional.vector_quantization import quantize_residual_vector

__all__ = ["ResidualVectorQuantizer"]


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer used in SoundStream.

    Args:
        codebook_size (int): Size of codebook.
        embedding_dim (int): Number of embedding dimensions.
        num_layers (int): Number of residual layers.

    """

    def __init__(self, codebook_size: int, embedding_dim: int, num_layers: int) -> None:
        super().__init__()

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        codebooks = []

        for _ in range(num_layers):
            codebook = nn.Embedding(
                num_embeddings=codebook_size,
                embedding_dim=embedding_dim,
            )
            codebooks.append(codebook)

        self.codebooks = nn.ModuleList(codebooks)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Forward pass of vector quantizer.

        Args:
            input (torch.Tensor): Latent feature of shape (batch_size, embedding_dim, *).

        Returns:
            tuple: Tuple containing:

                - torch.Tensor: Selected embeddings of same shape as input.
                - torch.LongTensor: Indices of indices in codebook of shape
                    (batch_size, num_layers, *).

        """
        weight = []

        for codebook in self.codebooks:
            codebook: nn.Embedding
            weight.append(codebook.weight)

        output, indices = quantize_residual_vector(input, weight)

        return output, indices
