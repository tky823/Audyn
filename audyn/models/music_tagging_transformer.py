import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.music_tagging_transformer import (
    MusicTaggingTransformerEncoder,
    PositionalPatchEmbedding,
)
from .ast import BaseAudioSpectrogramTransformer

__all__ = [
    "MusicTaggingTransformer",
]


class MusicTaggingTransformer(BaseAudioSpectrogramTransformer):
    """Music Tagging Transformer."""

    def __init__(
        self,
        embedding: PositionalPatchEmbedding,
        backbone: MusicTaggingTransformerEncoder,
        aggregator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(embedding, backbone)

        self.aggregator = aggregator
        self.head = head

        if self.aggregator is None and self.head is not None:
            warnings.warn(
                "Head is given, but aggregator is not given, "
                "which may lead to unexpected behavior.",
                UserWarning,
                stacklevel=2,
            )

    def forward(
        self,
        input: torch.Tensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        input = self.pad_by_length(input, length=length)
        x = self.embedding(input)
        padding_mask = self.compute_padding_mask(input, length=length)
        x = self.transformer_forward(x, padding_mask=padding_mask)

        if self.aggregator is not None:
            x = self.aggregator(x)

        if self.head is not None:
            x = self.head(x)

        output = x

        return output

    def compute_padding_mask(
        self,
        input: torch.Tensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        if length is None:
            padding_mask = None
        else:
            factory_kwargs = {
                "dtype": torch.long,
                "device": length.device,
            }
            _, n_bins, max_frames = input.size()
            width = []

            for _length in length:
                n_frames = _length.item()
                _width = self.embedding.compute_output_shape(n_bins, n_frames)
                width.append(_width)

            width = torch.tensor(width, **factory_kwargs)
            max_width = self.embedding.compute_output_shape(n_bins, max_frames)
            padding_mask = torch.arange(max_width, **factory_kwargs) >= width.unsqueeze(dim=-1)

            num_head_tokens = 0

            if self.embedding.insert_cls_token:
                num_head_tokens += 1

            if self.embedding.insert_dist_token:
                num_head_tokens += 1

            padding_mask = F.pad(padding_mask, (num_head_tokens, 0), value=False)

        return padding_mask

    @classmethod
    def build_from_default_config(cls) -> "MusicTaggingTransformer":
        pass

    @classmethod
    def build_from_pretrained(cls) -> "MusicTaggingTransformer":
        pass
