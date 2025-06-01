import warnings
from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from ..modules.vit import PatchEmbedding

__all__ = [
    "OpenAICLIPImageEncoder",
    "Aggregator",
    "HeadTokensAggregator",
    "Head",
    "LinearHead",
]


class _CLIPImageEncoder(nn.Module):
    """Base class of CLIP image encoder.

    Args:
        embedding (audyn.modules.vit.PositionalPatchEmbedding): Patch embedding
            followed by positional embedding.
        backbone (nn.TransformerEncoder): Transformer (encoder).

    """

    def __init__(
        self,
        embedding: PatchEmbedding,
        backbone: nn.TransformerEncoder,
        aggregator: Optional["Aggregator"] = None,
        head: Optional["Head"] = None,
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.backbone = backbone
        self.aggregator = aggregator
        self.head = head

        assert not embedding.insert_cls_token, "[CLS] token is not supported."
        assert not embedding.insert_dist_token, "[DST] token is not supported."

        if self.aggregator is None and self.head is not None:
            warnings.warn(
                "Head is given, but aggregator is not given, "
                "which may lead to unexpected behavior.",
                UserWarning,
                stacklevel=2,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of _CLIPImageEncoder.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, in_channels, n_bins, n_frames),
                where in_channels corresponds to number of fused chunks.

        Returns:
            torch.Tensor: (batch_size, num_downsampled_patches, embedding_dim), where
                ``num_downsampled_patches`` corresponds to
                ``downsampled_height * downsampled_width``.

        """
        x = self.embedding(input)
        output = self.backbone(x)

        if self.aggregator is not None:
            output = self.aggregator(output)

        if self.head is not None:
            output = self.head(output)

        return output


class Aggregator(nn.Module):
    """Base class of module to aggregate features."""

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of Aggregator.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embedding_dim).
            padding_mask (torch.BoolTensor, optional): Padding mask of shape (batch_size, length).

        Returns:
            torch.Tensor: Aggregated feature of shape (batch_size, embedding_dim).

        """
        pass


class HeadTokensAggregator(Aggregator):
    """Module of aggregation by extraction of head tokens.

    Args:
        insert_cls_token (bool): Given sequence is assumed to contain [CLS] token.
        insert_dist_token (bool): Given sequence is assumed to contain [DIST] token.

    """

    def __init__(
        self,
        insert_cls_token: bool = True,
        insert_dist_token: bool = True,
    ) -> None:
        super().__init__()

        if not insert_cls_token and not insert_dist_token:
            raise ValueError(
                "At least one of insert_cls_token and insert_dist_token should be True."
            )

        self.insert_cls_token = insert_cls_token
        self.insert_dist_token = insert_dist_token

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of HeadTokensAggregator.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embedding_dim).
            padding_mask (torch.BoolTensor, optional): Padding mask of shape (batch_size, length).

        Returns:
            torch.Tensor: Aggregated feature of shape (batch_size, embedding_dim).

        .. note::

            padding_mask is ignored.

        """
        num_head_tokens = 0

        if self.insert_cls_token:
            num_head_tokens += 1

        if self.insert_dist_token:
            num_head_tokens += 1

        head_tokens, _ = torch.split(
            input, [num_head_tokens, input.size(-2) - num_head_tokens], dim=-2
        )
        output = torch.mean(head_tokens, dim=-2)

        return output

    def extra_repr(self) -> str:
        s = []

        if self.insert_cls_token:
            s.append("cls_token=True")

        if self.insert_dist_token:
            s.append("dist_token=True")

        s = ", ".join(s)

        return s


class Head(nn.Module):
    """Base class of Head module to transform aggregated feature."""

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass


class LinearHead(Head):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLPHead.

        Args:
            input (torch.Tensor): Aggregated feature of shape (batch_size, in_channels).

        Returns:
            torch.Tensor: Transformed feature of shape (batch_size, out_channels).

        """
        output = self.linear(input)

        return output


class OpenAICLIPImageEncoder(_CLIPImageEncoder):
    """Wrapper class of _CLIPImageEncoder."""
