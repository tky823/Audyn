from typing import Optional

import torch
import torch.nn as nn

from .glow import ActNorm1d

__all__ = ["MaskedLayerNorm", "ActNorm1d"]


class MaskedLayerNorm(nn.LayerNorm):
    """Layer normalization with masking."""

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of MaskedLayerNorm.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, *).
            padding_mask (torch.BoolTensor, optional): Padding mask.
                Shape should be broadcastable to shape of input.

        Returns:
            torch.Tensor: Output feature of same shape as input.

        """
        if padding_mask is None:
            return super().forward(input)
        else:
            normalized_shape = self.normalized_shape
            weight = self.weight
            bias = self.bias
            eps = self.eps

            x = input.masked_fill(padding_mask, 0)
            normalized_dim = tuple(range(-1, -len(normalized_shape) - 1, -1))

            expanded_padding_mask = padding_mask.expand(x.size())
            expanded_non_padding_mask = torch.logical_not(expanded_padding_mask)
            num_elements = expanded_non_padding_mask.sum(dim=normalized_dim, keepdim=True)

            # to avoid zero-division
            num_elements = num_elements.masked_fill(padding_mask, 1)
            mean = torch.sum(x, dim=normalized_dim, keepdim=True) / num_elements
            squared_dev = torch.masked_fill((x - mean) ** 2, padding_mask, 0)
            var = squared_dev.sum(dim=normalized_dim, keepdim=True) / num_elements
            std = torch.sqrt(var + eps)
            x = (x - mean) / std
            x = weight * x + bias
            output = x.masked_fill(padding_mask, 0)

            return output
