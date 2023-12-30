"""Vector quantization modules."""
from typing import Tuple

import torch
import torch.nn as nn

from ..functional.vector_quantization import quantize_vector

__all__ = ["VectorQuantizer"]


class VectorQuantizer(nn.Module):
    """Vector quantizer used in VQVAE.

    Args:
        codebook_size (int): Size of codebook.
        embedding_dim (int): Number of embedding dimensions.
        init_by_kmeans (int): Number of iterations in k-means clustering initialization.
            If non-positive value is given, k-means clustering initialization is not used.
        seed (int): Random seed for k-means clustering initialization.

    """

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        init_by_kmeans: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()

        self.codebook = nn.Embedding(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim,
        )

        self.seed = seed
        self.init_by_kmeans = init_by_kmeans

        if self.init_by_kmeans > 0:
            self.register_parameter(
                "is_initialized", nn.Parameter(torch.tensor(False), requires_grad=False)
            )
        else:
            self.register_parameter(
                "is_initialized", nn.Parameter(torch.tensor(True), requires_grad=False)
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
        self.is_initialized: torch.Tensor
        is_initialized: bool = self.is_initialized.item()

        if not is_initialized:
            self._initialize_parameters(input)
            self.is_initialized.fill_(True)

        output, indices = quantize_vector(input, self.codebook.weight)

        return output, indices

    @torch.no_grad()
    def _initialize_parameters(self, encoded: torch.Tensor) -> None:
        # TODO: support DDP
        self.is_initialized: torch.Tensor
        is_initialized: bool = self.is_initialized.item()

        assert self.init_by_kmeans > 0 and not is_initialized

        g = torch.Generator(device=encoded.device)
        g.manual_seed(self.seed)

        batch_size, embedding_dim, *_ = encoded.size()
        encoded = encoded.view(batch_size, embedding_dim, -1)
        encoded = encoded.permute(0, 2, 1).contiguous()
        encoded = encoded.view(-1, embedding_dim)

        # select ``codebook_size`` embeddings from encoded features
        codebook_size = self.codebook.weight.size(0)
        indices = torch.randperm(encoded.size(0), generator=g, dtype=torch.long)
        indices = indices[:codebook_size]
        centroids = encoded[indices]

        for _ in range(self.init_by_kmeans):
            norm = torch.sum(centroids**2, dim=-1)
            dot = torch.matmul(encoded, centroids.transpose(1, 0))
            distance = norm - 2 * dot
            indices = torch.argmin(distance, dim=-1)
            eye = torch.eye(codebook_size, device=encoded.device, dtype=encoded.dtype)
            assignments = eye[indices]
            assignments = assignments.permute(1, 0)
            num_assignments = assignments.sum(dim=-1, keepdim=True)
            centroids = torch.matmul(assignments, encoded) / num_assignments

        self.codebook.weight.data.copy_(centroids)
