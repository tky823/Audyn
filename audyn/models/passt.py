from typing import Optional

import torch
import torch.nn as nn

from ..modules.passt import DisentangledPositionalPatchEmbedding, Patchout
from .ast import Aggregator, BaseAudioSpectrogramTransformer, Head

__all__ = [
    "PaSST",
    "DisentangledPositionalPatchEmbedding",
    "Patchout",
]


class PaSST(BaseAudioSpectrogramTransformer):
    """Patchout faSt Spectrogram Transformer (PaSST).

    Args:
        embedding (audyn.models.passt.DisentangledPositionalPatchEmbedding): Patch embedding
            followed by positional embeddings disentangled by frequency and time ones.
        dropout (audyn.models.passt.PatchDropout): Patch dropout module. The expected input
            is 4D feature (batch_size, embedding_dim, height, width). The expected output is
            tuple of 3D feature (batch_size, max_length, embedding_dim) and length (batch_size,).
        backbone (nn.TransformerEncoder): Transformer (encoder).

    """

    def __init__(
        self,
        embedding: DisentangledPositionalPatchEmbedding,
        dropout: Patchout,
        backbone: nn.TransformerEncoder,
        aggregator: Optional[Aggregator] = None,
        head: Optional[Head] = None,
    ) -> None:
        super(BaseAudioSpectrogramTransformer, self).__init__()

        self.embedding = embedding
        self.dropout = dropout
        self.backbone = backbone
        self.aggregator = aggregator
        self.head = head

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input)
        x_patch = self.spectrogram_to_patches(input)

        _, _, height, width = x_patch.size()

        head_tokens, x = self.split_sequence(x)
        x = self.sequence_to_patches(x, height=height, width=width)
        x, _ = self.dropout(x)

        assert (
            x.dim() == 3
        ), "Return of dropout should be 3D (batch_size, max_length, embedding_dim)."

        x = self.prepend_tokens(x, tokens=head_tokens)
        output = self.transformer_forward(x)

        if self.aggregator is not None:
            output = self.aggregator(output)

        if self.head is not None:
            output = self.head(output)

        return output
