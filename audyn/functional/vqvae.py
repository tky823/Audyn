from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = ["quantize_vector"]


def quantize_vector(
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

    assert n_dims > 2, "n_dims is expected to be (batch_size, embedding_dim, *)."

    batch_size, embedding_dim, *shape = input.size()

    with torch.no_grad():
        z_e = input.view(batch_size, embedding_dim, -1)
        z_e = z_e.permute(1, 0, 2).contiguous()
        z_e = z_e.view(embedding_dim, -1)
        e = weight.view(-1, embedding_dim, 1)
        distance = torch.sum((z_e - e) ** 2, dim=1)
        indices = torch.argmin(distance, dim=0)

    z_q = F.embedding(indices, weight)
    z_q = z_q.view(batch_size, -1, embedding_dim)
    z_q = z_q.permute(0, 2, 1).contiguous()
    output = z_q.view(batch_size, embedding_dim, *shape)

    indices = indices.view(batch_size, *shape)

    return output, indices
