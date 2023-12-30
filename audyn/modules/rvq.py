from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ..functional.vector_quantization import quantize_residual_vector, quantize_vector

__all__ = ["ResidualVectorQuantizer"]


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer used in SoundStream.

    Args:
        codebook_size (int): Size of codebook.
        embedding_dim (int): Number of embedding dimensions.
        num_stages (int): Number of residual steps.
        dropout (bool): Dropout of RVQ. Default: ``True``.
        init_by_kmeans (int): Number of iterations in k-means clustering initialization.
            If non-positive value is given, k-means clustering initialization is not used.
        seed (int): Random seed for k-means clustering initialization.

    """

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        num_stages: int,
        dropout: bool = True,
        init_by_kmeans: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.num_stages = num_stages
        self.dropout = dropout

        codebooks = []

        for _ in range(num_stages):
            codebook = nn.Embedding(
                num_embeddings=codebook_size,
                embedding_dim=embedding_dim,
            )
            codebooks.append(codebook)

        self.codebooks = nn.ModuleList(codebooks)

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
                - torch.LongTensor: Indices of codebook of shape (batch_size, num_stages', *),
                    where num_stages' might be changed if ``dropout=True``.
                    To disable this feature, set ``dropout=False`` or call ``.eval()``.

        """
        self.is_initialized: torch.Tensor
        is_initialized: bool = self.is_initialized.item()

        if not is_initialized:
            self._initialize_parameters(input)
            self.is_initialized.fill_(True)

        if self.dropout and self.training:
            num_stages = torch.randint(0, len(self.codebooks), ()) + 1

            if dist.is_available() and dist.is_initialized():
                # gather gathered_num_stages
                # gathered_num_stages: () -> (num_gpus,)
                gathered_num_stages = [
                    torch.zeros_like(num_stages) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_num_stages, gathered_num_stages)

                # use num_stages on 1st GPU
                num_stages = gathered_num_stages[0]

            num_stages = num_stages.item()
        else:
            num_stages = len(self.codebooks)

        weight = []

        for codebook in self.codebooks[:num_stages]:
            codebook: nn.Embedding
            weight.append(codebook.weight)

        output, indices = quantize_residual_vector(input, weight)

        return output, indices

    @torch.no_grad()
    def _initialize_parameters(self, encoded: torch.Tensor) -> None:
        self.is_initialized: torch.Tensor
        is_initialized: bool = self.is_initialized.item()
        is_distributed = dist.is_available() and dist.is_initialized()

        assert self.init_by_kmeans > 0 and not is_initialized

        if is_distributed:
            # TODO: support DDP
            raise NotImplementedError("DDP is not supported.")

        g = torch.Generator(device=encoded.device)
        g.manual_seed(self.seed)

        batch_size, embedding_dim, *_ = encoded.size()
        encoded = encoded.view(batch_size, embedding_dim, -1)
        encoded = encoded.permute(0, 2, 1).contiguous()
        encoded = encoded.view(-1, embedding_dim)
        residual = encoded
        reconstructed = 0

        for codebook in self.codebooks:
            # select ``codebook_size`` embeddings from encoded features
            codebook: nn.Embedding
            codebook_size = codebook.weight.size(0)
            indices = torch.randperm(
                residual.size(0),
                generator=g,
                device=residual.device,
                dtype=torch.long,
            )
            indices = indices[:codebook_size]
            centroids = residual[indices]

            with autocast(enabled=False):
                for _ in range(self.init_by_kmeans):
                    norm = torch.sum(centroids**2, dim=-1)
                    dot = torch.matmul(residual, centroids.transpose(1, 0))
                    distance = norm - 2 * dot
                    indices = torch.argmin(distance, dim=-1)
                    unique_indices = torch.unique(indices)
                    num_drops = codebook_size - unique_indices.size(0)

                    if num_drops > 0:
                        index_counts = torch.bincount(indices, minlength=codebook_size)
                        least_used_indices = torch.argsort(index_counts)
                        most_used_indices = least_used_indices[num_drops:]
                        least_used_indices = least_used_indices[:num_drops]
                        unused_centroids = centroids[least_used_indices]
                        assignments = F.one_hot(indices, num_classes=codebook_size)
                        assignments = assignments.permute(1, 0)
                        num_assignments = assignments.sum(dim=-1, keepdim=True)
                        prod = torch.matmul(assignments.to(residual.dtype), residual)
                        prod = prod[most_used_indices]
                        num_assignments = num_assignments[most_used_indices]
                        used_centroids = prod / num_assignments.to(residual.dtype)
                        centroids = torch.cat([used_centroids, unused_centroids], dim=0)
                    else:
                        assignments = F.one_hot(indices, num_classes=codebook_size)
                        assignments = assignments.permute(1, 0)
                        prod = torch.matmul(assignments.to(residual.dtype), residual)
                        num_assignments = assignments.sum(dim=-1, keepdim=True)
                        centroids = prod / num_assignments.to(residual.dtype)

                codebook.weight.data.copy_(centroids)

                output, _ = quantize_vector(residual, codebook.weight)
                reconstructed = reconstructed + output
                residual = encoded - reconstructed
