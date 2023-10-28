from typing import Tuple

import torch
import torch.nn as nn

from ..modules.vqvae import VectorQuantizer

__all__ = ["VQVAE"]


class VQVAE(nn.Module):
    """Vector quantized-variational autoencoder.

    Args:
        encoder (nn.Module): Encoder which returns latent feature of
            shape (batch_size, embedding_dim, *).
        decoder (nn.Module): Decoder which takes latent feature of
            shape (batch_size, embedding_dim, *).
        codebook_size (int): Size of codebook.
        embedding_dim (int): Number of embedding dimension.

    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        codebook_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.vector_quantizer = VectorQuantizer(codebook_size, embedding_dim)
        self.decoder = decoder

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Forward pass of VQVAE.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, *input_shape).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Reconstructed feature of same shape as input.
                - torch.Tensor: Latent feature of shape \
                    (batch_size, embedding_dim, *latent_shape). In most cases, latent_shape is \
                    smaller than input_shape.
                - torch.Tensor: Quantized latent feature. The shape is same as latent feature.
                - torch.Tensor: Indices of embeddings in codebook.

        .. note::

            Gradient from decoder does not back propagate to codebook.

        """
        encoded = self.encoder(input)
        quantized, indices = self.vector_quantizer(encoded)
        quantized_straight_through = encoded + torch.detach(quantized - encoded)
        output = self.decoder(quantized_straight_through)

        return output, encoded, quantized, indices

    @torch.no_grad()
    def inference(self, quantized: torch.Tensor) -> torch.Tensor:
        """Inference of VQVAE.

        Args:
            quantized (torch.Tensor): Following two types are supported.
                1. Quantized latent feature (torch.FloatTensor).
                2. Indices of quantized latent feature (torch.LongTensor).

        Returns:
            torch.Tensor: Reconstructed feature.

        """
        if torch.is_floating_point(quantized):
            pass
        elif quantized.dtype in [torch.long]:
            quantized = self.vector_quantizer.codebook(quantized)
            batch_size, *shape, embedding_dim = quantized.size()
            quantized = quantized.view(batch_size, -1, embedding_dim)
            quantized = quantized.permute(0, 2, 1).contiguous()
            quantized = quantized.view(batch_size, embedding_dim, *shape)
        else:
            raise TypeError("Invalid dtype {} is given.".format(quantized.dtype))

        output = self.decoder(quantized)

        return output
