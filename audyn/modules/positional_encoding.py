from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "AbsolutePositionalEncoding",
    "RotaryPositionalEmbedding",
    "RoPE",
]


class AbsolutePositionalEncoding(nn.Module):
    def __init__(
        self,
        base: int = 10000,
        batch_first: bool = False,
    ) -> None:
        super().__init__()

        self.base = base
        self.batch_first = batch_first

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of absolute positional encoding.

        Args:
            input (torch.Tensor): Input embeddings of shape (batch_size, length, num_features)
                if ``batch_first=True``, otherwise (length, batch_size, num_features).

        Returns:
            torch.Tensor: Output embeddings whose shape is same as input.

        """
        base = self.base
        batch_first = self.batch_first

        if batch_first:
            _, length, num_features = input.size()
        else:
            length, _, num_features = input.size()

        assert num_features % 2 == 0, "Feature dimension is expected to be even number."

        pos_seq = torch.arange(length)
        num_seq = torch.arange(0, num_features, 2) / num_features
        theta = pos_seq.unsqueeze(dim=-1) / (base**num_seq)

        sin = torch.sin(theta)
        cos = torch.cos(theta)

        pos_emb = torch.stack([sin, cos], dim=2)  # (length, num_features // 2, 2)
        pos_emb = pos_emb.view(length, num_features)
        pos_emb = pos_emb.to(input.device)

        if not batch_first:
            pos_emb = pos_emb.unsqueeze(dim=1)

        output = input + pos_emb

        return output


class RotaryPositionalEmbedding(nn.Module):
    """RoPE: Rotary positional embedding proposed in [#su2024roformer]_.

    .. [#su2024roformer]
        J. Su et al., "Roformer: Enhanced transformer with rotary position embedding,"
        *Neurocomputing*, vol. 568, 2024.

    """

    def __init__(
        self,
        embed_dim: int,
        batch_first: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        super().__init__()

        if embed_dim % 2 != 0:
            raise ValueError("embed_dim should be even number.")

        self.embed = nn.Parameter(
            torch.empty(
                embed_dim // 2,
                **factory_kwargs,
            ),
            requires_grad=True,
        )

        self.batch_first = batch_first

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of RoPE.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embed_dim)
                if ``batch_first=True``, otherwise (length, batch_size, embed_dim).

        Returns:
            torch.Tensor: Sequence of same shape as input.

        """
        batch_first = self.batch_first
        embed = self.embed

        device = input.device

        if batch_first:
            x_cos = input
        else:
            x_cos = input.transpose(1, 0)

        batch_size, length, embed_dim = x_cos.size()

        x_cos = x_cos.view(batch_size, length, embed_dim // 2, 2)
        x_sin_pre, x_sin_post = torch.unbind(x_cos, dim=-1)
        x_sin = torch.stack([-x_sin_post, x_sin_pre], dim=-1)

        positions = torch.arange(length, dtype=torch.long, device=device)
        theta = positions.unsqueeze(dim=-1) * embed
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        x = x_cos * cos_theta.unsqueeze(dim=-1) + x_sin * sin_theta.unsqueeze(dim=-1)
        x = x.view(batch_size, length, embed_dim)

        if batch_first:
            output = x
        else:
            output = x.transpose(1, 0)

        return output


class RoPE(RotaryPositionalEmbedding):
    """Alias of RotaryPositionalEmbedding."""
