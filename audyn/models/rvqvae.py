from typing import Tuple, overload

import torch
import torch.nn as nn

from ..modules.rvq import ResidualVectorQuantizer
from .vae import BaseVAE

__all__ = ["RVQVAE"]


class RVQVAE(BaseVAE):
    @overload
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        codebook_size: int,
        embedding_dim: int,
        num_stages: int,
        dropout: bool = True,
        init_by_kmeans: int = 0,
        seed: int = 0,
    ) -> None:
        """Residual vector quantized-variational autoencoder.

        Args:
            encoder (nn.Module): Encoder which returns latent feature of
                shape (batch_size, embedding_dim, *).
            decoder (nn.Module): Decoder which takes latent feature of
                shape (batch_size, embedding_dim, *).
            codebook_size (int): Size of codebook.
            embedding_dim (int): Number of embedding dimension.
            num_stages (int): Number of stages of RVQ.
            dropout (bool): Dropout of RVQ. Default: ``True``.
            init_by_kmeans (int): Number of iterations in k-means clustering initialization.
                If non-positive value is given, k-means clustering initialization is not used.
            seed (int): Random seed for k-means clustering initialization.

        """

    @overload
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        vector_quantizer: ResidualVectorQuantizer,
    ) -> None:
        """Residual vector quantized-variational autoencoder.

        Args:
            encoder (nn.Module): Encoder which returns latent feature of
                shape (batch_size, embedding_dim, *).
            decoder (nn.Module): Decoder which takes latent feature of
                shape (batch_size, embedding_dim, *).
            vector_quantizer (int): Residual vector quantizer.

        """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        if len(args) + len(kwargs) == 1:
            if len(args) == 1:
                (vector_quantizer,) = args

                assert isinstance(vector_quantizer, nn.Module), (
                    "nn.Module is required as positional argument."
                )
            else:
                assert "vector_quantizer" in kwargs, (
                    "vector_quantizer is required as keyword argument."
                )

                (vector_quantizer,) = kwargs["vector_quantizer"]

                assert isinstance(vector_quantizer, nn.Module), (
                    "nn.Module is required as vector_quantizer."
                )
        else:
            required_keys = ["codebook_size", "embedding_dim"]
            optional_keys = ["num_stages", "dropout", "init_by_kmeans", "seed"]
            keys = required_keys + optional_keys
            args_keys = keys[: len(args)]
            vector_quantizer_kwargs = {}

            for key, arg in zip(args_keys, args):
                vector_quantizer_kwargs[key] = arg

            for key, kwarg in kwargs.items():
                assert key not in vector_quantizer_kwargs
                assert key in keys[len(args) :]

                vector_quantizer_kwargs[key] = kwarg

            vector_quantizer = ResidualVectorQuantizer(**vector_quantizer_kwargs)

        self.encoder = encoder
        self.vector_quantizer = vector_quantizer
        self.decoder = decoder

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Forward pass of RVQVAE.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, *input_shape).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Reconstructed feature of same shape as input.
                - torch.Tensor: Encoded latent feature of shape \
                    (batch_size, embedding_dim, *latent_shape). In most cases, latent_shape is \
                    smaller than input_shape.
                - torch.Tensor: Quantized latent feature of shape \
                    (batch_size, num_stages, embedding_dim, *latent_shape).
                - torch.Tensor: Residual latent feature of shape \
                    (batch_size, num_stages, embedding_dim, *latent_shape).
                - torch.Tensor: Indices of embeddings in codebook of shape \
                    (batch_size, num_stages, *latent_shape).

        .. note::

            Gradient from decoder does not back propagate to codebook.

        """
        encoded = self.encode(input)
        hierarchical_quantized, hierarchical_residual, indices = self.quantize(encoded)
        quantized_straight_through = hierarchical_residual + torch.detach(
            hierarchical_quantized - hierarchical_residual
        )
        quantized_straight_through = quantized_straight_through.sum(dim=1)
        output = self.decode(quantized_straight_through, stage_wise=False)

        return output, encoded, hierarchical_quantized, hierarchical_residual, indices

    @torch.no_grad()
    def inference(self, quantized: torch.Tensor, stage_wise: bool = True) -> torch.Tensor:
        """Inference of RVQVAE.

        Args:
            quantized (torch.Tensor): Following two types are supported.
                1. Quantized latent feature of shape (batch_size, num_stages, *latent_shape)
                    or (batch_size, *latent_shape). dtype is torch.FloatTensor.
                2. Indices of quantized latent feature of shape
                    (batch_size, num_stages, *latent_shape). dtype is torch.LongTensor.
            stage_wise (bool): If ``True``, ``quantized`` has ``num_stages`` dimension at axis = 1.

        Returns:
            torch.Tensor: Reconstructed feature.

        """
        output = self.decode(quantized, stage_wise=stage_wise)

        return output

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        """Encode input.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Encoded feature of shape
                (batch_size, num_stages, embedding_dim, *latent_shape).

        """
        output = self.encoder(input)

        return output

    def decode(self, quantized: torch.Tensor, stage_wise: bool = True) -> torch.Tensor:
        """Decode quantized latent feature.

        Args:
            quantized (torch.Tensor): Following two types are supported.
                1. Quantized latent feature of shape (batch_size, num_stages, *latent_shape)
                    or (batch_size, *latent_shape). dtype is torch.FloatTensor.
                2. Indices of quantized latent feature of shape
                    (batch_size, num_stages, *latent_shape). dtype is torch.LongTensor.
            stage_wise (bool): If ``True``, ``quantized`` has ``num_stages`` dimension at axis = 1.

        Returns:
            torch.Tensor: Reconstructed feature.

        """
        if torch.is_floating_point(quantized):
            if stage_wise:
                # (batch_size, num_stages, *latent_shape) -> (batch_size, *latent_shape)
                quantized = quantized.sum(dim=1)
        elif quantized.dtype in [torch.long]:
            # to support torch.cuda.LongTensor, check dtype
            if not stage_wise:
                raise ValueError(
                    "Only stage_wise=True is supported when quantization indices are given."
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

    @torch.no_grad()
    def sample(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Non-differentiable sampling.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, *input_shape).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Quantized feature of shape \
                    (batch_size, num_stages, embedding_dim, *latent_shape). In most cases, \
                    latent_shape is smaller than input_shape.
                - torch.Tensor: Residual feature of shape \
                    (batch_size, num_stages, embedding_dim, *latent_shape).
                - torch.LongTensor: Quantization indices of shape \
                    (batch_size, num_stages, *latent_shape).

        .. note::

            This method does not use reparametrization trick.

        """
        encoded = self.encode(input)
        quantized, residual, indices = self.quantize(encoded)

        return quantized, residual, indices

    def rsample(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Differentiable sampling with straight through estimator.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, *input_shape).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Quantized latent feature of shape \
                    (batch_size, num_stages, embedding_dim, *latent_shape). In most cases, \
                    latent_shape is smaller than input_shape.
                - torch.Tensor: Residual latent feature of shape \
                    (batch_size, num_stages, embedding_dim, *latent_shape).
                - torch.LongTensor: Quantization indices of shape \
                    (batch_size, num_stages, *latent_shape).

        .. note::

            This method does not use reparametrization trick.

        """
        encoded = self.encode(input)
        quantized, residual, indices = self.quantize(encoded)
        quantized_straight_through = residual + torch.detach(quantized - residual)

        return quantized_straight_through, residual, indices

    def quantize(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Apply vector_quantizer.

        Args:
            input (torch.Tensor): Encoded feature of shape (batch_size, embedding_dim, *).

        Returns:
            tuple: Tuple of tensors containing:

                - torch.Tensor: Selected embeddings of shape
                    (batch_size, num_stages, embedding_dim, *).
                - torch.Tensor: Residual vectors of shape
                    (batch_size, num_stages, embedding_dim, *).
                - torch.LongTensor: Indices of indices in codebook of shape (batch_size, *).

        """
        quantized, residual, indices = self.vector_quantizer(input)

        return quantized, residual, indices
