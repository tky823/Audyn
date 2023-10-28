from typing import Tuple

import torch
import torch.nn as nn

from ..functional.vqvae import quantize_vector

__all__ = ["VectorQuantizer"]


class VectorQuantizer(nn.Module):
    """Vector quantizer used in VQVAE.

    Args:
        codebook_size (int): Size of codebook.
        embedding_dim (int): Number of embedding dimensions.

    """

    def __init__(self, codebook_size: int, embedding_dim: int) -> None:
        super().__init__()

        self.codebook = nn.Embedding(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim,
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Forward pass of vector quantizer.

        Args:
            input (torch.Tensor): Latent feature of shape (batch_size, embedding_dim, *).

        Returns:
            tuple: Tuple containing:

                - torch.Tensor: Selected embeddings of same shape as input.
                - torch.LongTensor: Indices of indices in codebook of shape (batch_size, *).

        """
        output, indices = quantize_vector(input, self.codebook.weight)

        return output, indices
