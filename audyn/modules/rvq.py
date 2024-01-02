from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast

from ..functional.vector_quantization import quantize_residual_vector, quantize_vector
from .vq import BaseVectorQuantizer

__all__ = ["ResidualVectorQuantizer"]


class ResidualVectorQuantizer(BaseVectorQuantizer):
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
        super().__init__(init_by_kmeans=init_by_kmeans, seed=seed)

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

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Forward pass of residual vector quantizer.

        Args:
            input (torch.Tensor): Latent feature of shape (batch_size, embedding_dim, *).

        Returns:
            tuple: Tuple containing:

                - torch.Tensor: Selected embeddings of same shape as input.
                - torch.LongTensor: Indices of codebook of shape (batch_size, num_stages', *),
                    where num_stages' might be changed if ``dropout=True``.
                    To disable this feature, set ``dropout=False`` or call ``.eval()``.

        """
        if self.training and not self.is_initialized:
            self._initialize_parameters(input)

        if self.dropout and self.training:
            num_stages = torch.randint(0, len(self.codebooks), (), device=input.device) + 1

            if dist.is_available() and dist.is_initialized():
                # gather num_stages: () -> (num_gpus,)
                gathered_num_stages = [
                    torch.zeros_like(num_stages) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_num_stages, num_stages)

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
        is_distributed = dist.is_available() and dist.is_initialized()

        assert self.init_by_kmeans > 0 and not self.is_initialized

        if is_distributed:
            # gather encoded
            # (batch_size, embedding_dim, *) -> (num_gpus * batch_size, embedding_dim, *)
            gathered_encoded = [torch.zeros_like(encoded) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_encoded, encoded)
            encoded = torch.concat(gathered_encoded, dim=0)

        g = torch.Generator(device=encoded.device)
        g.manual_seed(self.seed)

        batch_size, embedding_dim, *_ = encoded.size()
        encoded = encoded.view(batch_size, embedding_dim, -1)
        encoded = encoded.permute(0, 2, 1).contiguous()
        encoded = encoded.view(-1, embedding_dim)
        num_grids = encoded.size(0)
        reconstructed = 0

        with autocast(enabled=False):
            for codebook in self.codebooks:
                # select ``codebook_size`` embeddings from encoded features
                codebook: nn.Embedding
                codebook_size = codebook.weight.size(0)

                if num_grids < codebook_size:
                    msg = (
                        "Since number of grids given to RVQ is smaller than "
                        f"codebook size {codebook_size}, "
                        "we cannot apply k-means clustering initialization."
                    )
                    msg += " Please use larger batch size or smaller codebook size."

                    raise RuntimeError(msg)

                residual = encoded - reconstructed
                indices = torch.randperm(
                    num_grids,
                    generator=g,
                    device=residual.device,
                    dtype=torch.long,
                )
                indices = indices[:codebook_size]
                centroids = residual[indices]

                for _ in range(self.init_by_kmeans):
                    centroids = self._update_kmeans_centroids(residual, centroids)

                codebook.weight.data.copy_(centroids)
                quantized, _ = quantize_vector(residual, codebook.weight)
                reconstructed = reconstructed + quantized

        self.is_initialized = True
