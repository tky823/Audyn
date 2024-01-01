from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

__all__ = ["quantize_vector", "quantize_residual_vector"]


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

    assert n_dims > 1, "n_dims is expected to be (batch_size, embedding_dim, *)."

    batch_size, embedding_dim, *shape = input.size()

    with torch.no_grad():
        z_e = input.view(batch_size, embedding_dim, -1)
        z_e = z_e.permute(1, 0, 2).contiguous()
        z_e = z_e.view(embedding_dim, -1)
        e = weight.view(-1, embedding_dim)
        dot = torch.matmul(e, z_e)
        norm = torch.sum(e**2, dim=1)
        distance = norm.unsqueeze(dim=-1) - 2 * dot
        indices = torch.argmin(distance, dim=0)

    z_q = F.embedding(indices, weight)
    z_q = z_q.view(batch_size, -1, embedding_dim)
    z_q = z_q.permute(0, 2, 1).contiguous()
    output = z_q.view(batch_size, embedding_dim, *shape)

    indices = indices.view(batch_size, *shape)

    return output, indices


def quantize_residual_vector(
    input: torch.Tensor, weight: Union[torch.Tensor, List[torch.Tensor]]
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Apply vector quantization proposed in VQ-VAE.

    Args:
        input (torch.Tensor): Latent feature of shape (batch_size, embedding_dim, *).
        weight (torch.Tensor or list): Embeddings in codebooks. Following two types are supported.
            - Stacked codebooks of shape (num_layers, codebook_size, embedding_dim)
            - List of codebooks. Shape of each item is (codebook_size, embedding_dim).

    Returns:
        tuple: Tuple of tensors containing:

            - torch.Tensor: Quantized embeddings of shape \
                (batch_size, num_layers, embedding_dim, *).
            - torch.LongTensor: Indices of indices in codebook of shape \
            (batch_size, num_layers, *).

    """
    if isinstance(weight, torch.Tensor):
        n_dims = weight.dim()

        assert (
            n_dims == 3
        ), "Shape of weight is expected to be (num_layers, codebook_size, embedding_dim)."
    elif isinstance(weight, list):
        pass
    else:
        raise ValueError(f"Invalid type {type(weight)} is given as weight.")

    reconstructed = 0
    output = []
    indices = []

    for _weight in weight:
        residual = input - reconstructed
        _output, _indices = quantize_vector(residual, _weight)
        reconstructed = reconstructed + _output
        output.append(_output)
        indices.append(_indices)

    output = torch.stack(output, dim=1)
    indices = torch.stack(indices, dim=1)

    return output, indices
