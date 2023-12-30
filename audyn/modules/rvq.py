from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from ..functional.vector_quantization import quantize_residual_vector

__all__ = ["ResidualVectorQuantizer"]


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer used in SoundStream.

    Args:
        codebook_size (int): Size of codebook.
        embedding_dim (int): Number of embedding dimensions.
        num_stages (int): Number of residual steps.
        dropout (bool): Dropout of RVQ. Default: ``True``.

    """

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        num_stages: int,
        dropout: bool = True,
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
        if self.dropout and self.training:
            num_stages = torch.randint(0, len(self.codebooks), ()) + 1

            if dist.is_available() and dist.is_initialized():
                # gather gathered_num_stages
                # gathered_num_stages:
                #     () -> (num_gpus,)
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
