"""Vector quantization modules."""
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.modules.module import _IncompatibleKeys

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
            self.is_initialized = False
        else:
            self.is_initialized = True

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Forward pass of vector quantizer.

        Args:
            input (torch.Tensor): Latent feature of shape (batch_size, embedding_dim, *).

        Returns:
            tuple: Tuple containing:

                - torch.Tensor: Selected embeddings of same shape as input.
                - torch.LongTensor: Indices of indices in codebook of shape (batch_size, *).

        """
        if not self.is_initialized:
            self._initialize_parameters(input)
            self.is_initialized = True

        output, indices = quantize_vector(input, self.codebook.weight)

        return output, indices

    @torch.no_grad()
    def _initialize_parameters(self, encoded: torch.Tensor) -> None:
        is_distributed = dist.is_available() and dist.is_initialized()

        assert self.init_by_kmeans > 0 and not self.is_initialized

        if is_distributed:
            # TODO: support DDP
            raise NotImplementedError("DDP is not supported.")

        g = torch.Generator(device=encoded.device)
        g.manual_seed(self.seed)

        batch_size, embedding_dim, *_ = encoded.size()
        encoded = encoded.view(batch_size, embedding_dim, -1)
        encoded = encoded.permute(0, 2, 1).contiguous()
        encoded = encoded.view(-1, embedding_dim)

        # select ``codebook_size`` embeddings from encoded features
        codebook_size = self.codebook.weight.size(0)
        indices = torch.randperm(
            encoded.size(0),
            generator=g,
            device=encoded.device,
            dtype=torch.long,
        )
        indices = indices[:codebook_size]
        centroids = encoded[indices]

        with autocast(enabled=False):
            for _ in range(self.init_by_kmeans):
                norm = torch.sum(centroids**2, dim=-1)
                dot = torch.matmul(encoded, centroids.transpose(1, 0))
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
                    prod = torch.matmul(assignments.to(encoded.dtype), encoded)
                    prod = prod[most_used_indices]
                    num_assignments = num_assignments[most_used_indices]
                    used_centroids = prod / num_assignments.to(encoded.dtype)
                    centroids = torch.cat([used_centroids, unused_centroids], dim=0)
                else:
                    assignments = F.one_hot(indices, num_classes=codebook_size)
                    assignments = assignments.permute(1, 0)
                    prod = torch.matmul(assignments.to(encoded.dtype), encoded)
                    num_assignments = assignments.sum(dim=-1, keepdim=True)
                    centroids = prod / num_assignments.to(encoded.dtype)

            self.codebook.weight.data.copy_(centroids)

    def state_dict(
        self,
        destination: Dict[str, Any] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        """Return state_dict of module.

        .. note::

            Returned ``state_dict`` includes ``is_initialized`` flag.
            In terms of simplicity, registering ``is_initialized`` as boolean tensor
            is better, but it is incompatible with DDP.

        """
        state_dict = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        state_dict.update({prefix + "is_initialized": self.is_initialized})

        return state_dict

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ) -> _IncompatibleKeys:
        is_initialized_key = "is_initialized"

        if is_initialized_key in state_dict.keys():
            is_initialized = state_dict.pop(is_initialized_key)

            if isinstance(is_initialized, torch.Tensor):
                # for backward compatibility
                is_initialized = is_initialized.item()

            self.is_initialized = is_initialized

        return super().load_state_dict(state_dict, strict=strict)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> Any:
        is_initialized_key = prefix + "is_initialized"

        if is_initialized_key in state_dict.keys():
            is_initialized = state_dict.pop(is_initialized_key)

            if isinstance(is_initialized, torch.Tensor):
                # for backward compatibility
                is_initialized = is_initialized.item()

            self.is_initialized = is_initialized

        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
