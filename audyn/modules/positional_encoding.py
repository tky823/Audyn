import torch
import torch.nn as nn

from ..functional.positional_encoding import (
    extrapolatable_rotary_positional_embedding,
    partial_rotary_positional_embedding,
    rotary_positional_embedding,
)

__all__ = [
    "AbsolutePositionalEncoding",
    "RotaryPositionalEmbedding",
    "ExtrapolatablePositionalEmbedding",
    "PartialRotaryPositionalEmbedding",
    "RoPE",
    "XPos",
    "PartialRoPE",
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

        positional_embedding = torch.stack([sin, cos], dim=2)  # (length, num_features // 2, 2)
        positional_embedding = positional_embedding.view(length, num_features)
        positional_embedding = positional_embedding.to(input.device)

        if not batch_first:
            positional_embedding = positional_embedding.unsqueeze(dim=1)

        output = input + positional_embedding

        return output


class RotaryPositionalEmbedding(nn.Module):
    """RoPE: Rotary positional embedding proposed in [#su2021roformer]_.

    .. [#su2021roformer]
        J. Su et al., "RoFormer: Enhanced transformer with rotary position embedding,"
        *Neurocomputing*, vol. 568, 2024.

    """

    def __init__(
        self,
        base: int = 10000,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.base = base
        self.batch_first = batch_first

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of RoPE.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, num_features)
                if ``batch_first=True``, otherwise (length, batch_size, num_features).

        Returns:
            torch.Tensor: Sequence of same shape as input.

        """
        output = rotary_positional_embedding(input, base=self.base, batch_first=self.batch_first)

        return output


class RoPE(RotaryPositionalEmbedding):
    """Alias of RotaryPositionalEmbedding."""


class ExtrapolatablePositionalEmbedding(nn.Module):
    """Extrapolatable positional embedding proposed in [#sun2023length]_.

    Args:
        invert_decay (bool): If ``True``, decay is inverted.
        smooth (float): Smoothing factor.

    .. [#sun2023length]
        Y. Sun et al., "A length-extrapolatable transformer,"
        in *ACL*, 2023.

    """

    def __init__(
        self,
        invert_decay: bool,
        smooth: float = 0.4,
        base: int = 10000,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.invert_decay = invert_decay
        self.smooth = smooth
        self.base = base
        self.batch_first = batch_first

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of xPos.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, num_features)
                if ``batch_first=True``, otherwise (length, batch_size, num_features).

        Returns:
            torch.Tensor: Sequence of same shape as input.

        """
        output = extrapolatable_rotary_positional_embedding(
            input,
            invert_decay=self.invert_decay,
            smooth=self.smooth,
            base=self.base,
            batch_first=self.batch_first,
        )

        return output


class XPos(ExtrapolatablePositionalEmbedding):
    """Alias of ExtrapolatablePositionalEmbedding."""


class PartialRotaryPositionalEmbedding(nn.Module):
    """Partial RoPE proposed in [#khan2026fractional]_.

    Args:
        base (int): Base value for calculating frequencies. Default is 10000.
        fraction (float): Proportion of features (0.0 to 1.0) to which
            rotary transformation is applied. Default is 0.5.
        batch_first (bool): If True, input and output tensors are in
            (batch_size, length, num_features) format. Default is True.

    .. [#khan2026fractional]
        M. A. Khan et al., "Fractional Rotation, Full Potential? Investigating Performance and Convergence of Partial RoPE,"
        2026.

    """

    def __init__(
        self,
        base: int = 10000,
        fraction: float = 0.5,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.base = base
        self.fraction = fraction
        self.batch_first = batch_first

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of p-RoPE.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, num_features)
                if ``batch_first=True``, otherwise (length, batch_size, num_features).

        Returns:
            torch.Tensor: Sequence of same shape as input.

        """
        output = partial_rotary_positional_embedding(
            input, base=self.base, fraction=self.fraction, batch_first=self.batch_first
        )

        return output


class PartialRoPE(PartialRotaryPositionalEmbedding):
    """Alias of PartialRotaryPositionalEmbedding."""
