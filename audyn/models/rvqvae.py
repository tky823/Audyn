from typing import Tuple, Union

import torch
import torch.nn as nn

from ..modules.rvq import ResidualVectorQuantizer
from .vae import BaseVAE

__all__ = ["RVQVAE"]


class RVQVAE(BaseVAE):
    """Residual vector quantized-variational autoencoder.

    Args:
        encoder (nn.Module): Encoder which returns latent feature of
            shape (batch_size, embedding_dim, *).
        decoder (nn.Module): Decoder which takes latent feature of
            shape (batch_size, embedding_dim, *).
        codebook_size (int): Size of codebook.
        embedding_dim (int): Number of embedding dimension.
        num_layers (int): Number of layers of RVQ.

    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        codebook_size: int,
        embedding_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.vector_quantizer = ResidualVectorQuantizer(
            codebook_size, embedding_dim, num_layers=num_layers
        )
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
                - torch.Tensor: Quantized latent feature of shape \
                    (batch_size, num_layers, embedding_dim, *latent_shape).
                - torch.Tensor: Indices of embeddings in codebook of shape \
                    (batch_size, num_layers, *latent_shape).

        .. note::

            Gradient from decoder does not back propagate to codebook.

        """
        encoded = self.encode(input, quantization=False)
        hierarchical_quantized, indices = self.vector_quantizer(encoded)
        quantized = hierarchical_quantized.sum(dim=1)
        quantized_straight_through = encoded + torch.detach(quantized - encoded)
        output = self.decode(quantized_straight_through, layer_wise=False)

        return output, encoded, hierarchical_quantized, indices

    @torch.no_grad()
    def inference(self, quantized: torch.Tensor, layer_wise: bool = True) -> torch.Tensor:
        """Inference of VQVAE.

        Args:
            quantized (torch.Tensor): Following two types are supported.
                1. Quantized latent feature of shape (batch_size, num_layers, *latent_shape)
                    or (batch_size, *latent_shape). dtype is torch.FloatTensor.
                2. Indices of quantized latent feature of shape
                    (batch_size, num_layers, *latent_shape). dtype is torch.LongTensor.
            layer_wise (bool): If ``True``, ``quantized`` has ``num_layers`` dimension at axis = 1.

        Returns:
            torch.Tensor: Reconstructed feature.

        """
        output = self.decode(quantized, layer_wise=layer_wise)

        return output

    def encode(
        self,
        input: torch.Tensor,
        quantization: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.LongTensor], torch.Tensor]:
        """Encode input.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, *input_shape).
            quantization (bool): If ``True``, quantization is applied.

        Returns:
            Output depends on ``quantization`` flag.
            If ``quantization=True``, tuple of quantized feature
            (batch_size, num_layers, embedding_dim, *latent_shape) and its indices
            (batch_size, num_layers, embedding_dim, *latent_shape) are returned.
            Otherwise, encoded feature (batch_size, num_layers, embedding_dim, *latent_shape)
            is returned.

        """
        encoded = self.encoder(input)

        if quantization:
            quantized, indices = self.vector_quantizer(encoded)

            return quantized, indices
        else:
            return encoded

    def decode(self, quantized: torch.Tensor, layer_wise: bool = True) -> torch.Tensor:
        """Decode quantized latent feature.

        Args:
            quantized (torch.Tensor): Following two types are supported.
                1. Quantized latent feature of shape (batch_size, num_layers, *latent_shape)
                    or (batch_size, *latent_shape). dtype is torch.FloatTensor.
                2. Indices of quantized latent feature of shape
                    (batch_size, num_layers, *latent_shape). dtype is torch.LongTensor.
            layer_wise (bool): If ``True``, ``quantized`` has ``num_layers`` dimension at axis = 1.

        Returns:
            torch.Tensor: Reconstructed feature.

        """
        if torch.is_floating_point(quantized):
            if layer_wise:
                # (batch_size, num_layers, *latent_shape) -> (batch_size, *latent_shape)
                quantized = quantized.sum(dim=1)
        elif quantized.dtype in [torch.long]:
            # to support torch.cuda.LongTensor, check dtype
            if not layer_wise:
                raise ValueError(
                    "Only layer_wise=True is supported when quantization indices are given."
                )

            quantized = torch.unbind(quantized, dim=1)
            stack_quantized = []

            for codebook, _quantized in zip(self.vector_quantizer.codebooks, quantized):
                _quantized = codebook(_quantized)
                stack_quantized.append(_quantized)

            stack_quantized = torch.stack(stack_quantized, dim=1)
            quantized = stack_quantized.sum(dim=1)
            batch_size, *shape, embedding_dim = quantized.size()
            quantized = quantized.view(batch_size, -1, embedding_dim)
            quantized = quantized.permute(0, 2, 1).contiguous()
            quantized = quantized.view(batch_size, embedding_dim, *shape)
        else:
            raise TypeError("Invalid dtype {} is given.".format(quantized.dtype))

        output = self.decoder(quantized)

        return output
