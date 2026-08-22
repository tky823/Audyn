from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.musicfm import Masker
from ..modules.vq import VectorQuantizer as _VectorQuantizer


class VectorQuantizer(_VectorQuantizer):
    """Vector quantizer used in MuQ-RVQ.

    Args:
        codebook_size (int): Size of codebook.
        embedding_dim (int): Number of embedding dimensions.
        init_by_kmeans (int): Number of iterations in k-means clustering initialization.
            If non-positive value is given, k-means clustering initialization is not used.
        seed (int): Random seed for k-means clustering initialization.

    .. note::

        Unlike ``VectorQuantizer``, this module selects codebooks
        based on cosine similarity.

    """

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Forward pass of vector quantizer.

        Args:
            input (torch.Tensor): Latent feature of shape (batch_size, embedding_dim, *).

        Returns:
            tuple: Tuple containing:

                - torch.Tensor: Selected embeddings of same shape as input.
                - torch.LongTensor: Indices of indices in codebook of shape (batch_size, *).

        """
        if self.training and not self.is_initialized:
            self._initialize_parameters(input)

        output, indices = _quantize_vector(input, self.codebook.weight)

        return output, indices


class MultiMasker(Masker):
    def __init__(
        self,
        mask_rate: float,
        num_stages: int,
        window_size: int = 4,
        noise_scale: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__(
            mask_rate,
            window_size=window_size,
            noise_scale=noise_scale,
            seed=seed,
        )

        self.num_stages = num_stages

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.BoolTensor]:
        output, mask = super().forward(input)

        _, num_frams = mask.size()

        mask = mask.unsqueeze(dim=-2)
        mask = mask.expand(-1, self.num_stages, num_frams)

        return output, mask


class MultiLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_stages: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__(in_features, num_stages * out_features, bias=bias, **factory_kwargs)

        self.num_stages = num_stages

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        num_stages = self.num_stages

        x = super().forward(input)

        batch_size, *shape, out_features = x.size()
        x = x.view(batch_size, -1, num_stages, out_features // num_stages)
        x = x.permute(0, 2, 1, 3)
        output = x.reshape(batch_size, num_stages, *shape, out_features // num_stages)

        return output


def _quantize_vector(
    input: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Apply vector quantization proposed in VQ-VAE.

    Args:
        input (torch.Tensor): Latent feature of shape (batch_size, embedding_dim, *).
        weight (torch.Tensor): Embeddings in codebook of shape
            (codebook_size, embedding_dim).

    Returns:
        tuple: Tuple containing:

            - torch.Tensor: Quantized embeddings of shape (batch_size, embedding_dim, *).
            - torch.LongTensor: Indices of indices in codebook of shape (batch_size, *).

    """
    n_dims = input.dim()

    assert n_dims > 1, "n_dims is expected to be (batch_size, embedding_dim, *)."

    batch_size, embedding_dim, *shape = input.size()

    with torch.no_grad():
        z_e = input.view(batch_size, embedding_dim, -1)
        z_e = z_e.permute(1, 0, 2).contiguous()
        z_e = z_e.view(embedding_dim, -1)
        e = weight.view(-1, embedding_dim)
        z_e = F.normalize(z_e, dim=0)
        e = F.normalize(e, dim=-1)
        similarity = torch.matmul(e, z_e)  # based on cosine similarity
        indices = torch.argmax(similarity, dim=0)

    z_q = F.embedding(indices, weight)
    z_q = z_q.view(batch_size, -1, embedding_dim)
    z_q = z_q.permute(0, 2, 1).contiguous()
    output = z_q.view(batch_size, embedding_dim, *shape)

    indices = indices.view(batch_size, *shape)

    return output, indices
