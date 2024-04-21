import torch
import torch.nn as nn

__all__ = [
    "AbsolutePositionalEncoding",
    "RotaryPositionalEmbedding",
    "ExtrapolatablePositionalEmbedding",
    "RoPE",
    "XPos",
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
        base = self.base
        batch_first = self.batch_first

        device = input.device

        if batch_first:
            x_cos = input
        else:
            x_cos = input.transpose(1, 0)

        batch_size, length, num_features = x_cos.size()

        x_cos = x_cos.view(batch_size, length, num_features // 2, 2)
        x_sin_pre, x_sin_post = torch.unbind(x_cos, dim=-1)
        x_sin = torch.stack([-x_sin_post, x_sin_pre], dim=-1)

        pos_seq = torch.arange(length)
        num_seq = torch.arange(0, num_features, 2) / num_features
        theta = pos_seq.unsqueeze(dim=-1) / (base**num_seq)

        sin = torch.sin(theta)
        cos = torch.cos(theta)
        sin = sin.to(device)
        cos = cos.to(device)

        x = x_sin * sin.unsqueeze(dim=-1) + x_cos * cos.unsqueeze(dim=-1)
        x = x.view(batch_size, length, num_features)

        if batch_first:
            output = x
        else:
            output = x.transpose(1, 0).contiguous()

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
        smooth = self.smooth
        base = self.base
        batch_first = self.batch_first

        device = input.device

        if batch_first:
            x_cos = input
        else:
            x_cos = input.transpose(1, 0)

        batch_size, length, num_features = x_cos.size()

        x_cos = x_cos.view(batch_size, length, num_features // 2, 2)
        x_sin_pre, x_sin_post = torch.unbind(x_cos, dim=-1)
        x_sin = torch.stack([-x_sin_post, x_sin_pre], dim=-1)

        pos_seq = torch.arange(length)
        num_seq = torch.arange(0, num_features, 2) / num_features
        theta = pos_seq.unsqueeze(dim=-1) / (base**num_seq)

        if self.invert_decay:
            decay = (1 + smooth) / (num_seq + smooth)
        else:
            decay = (num_seq + smooth) / (1 + smooth)

        decay = decay ** pos_seq.unsqueeze(dim=-1)

        sin = decay * torch.sin(theta)
        cos = decay * torch.cos(theta)
        sin = sin.to(device)
        cos = cos.to(device)
        x = x_sin * sin.unsqueeze(dim=-1) + x_cos * cos.unsqueeze(dim=-1)
        x = x.view(batch_size, length, num_features)

        if batch_first:
            output = x
        else:
            output = x.transpose(1, 0).contiguous()

        return output


class XPos(ExtrapolatablePositionalEmbedding):
    """Alias of ExtrapolatablePositionalEmbedding."""
