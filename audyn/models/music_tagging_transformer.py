import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.music_tagging_transformer import (
    MusicTaggingTransformerEncoder,
    PositionalPatchEmbedding,
)
from .ast import BaseAudioSpectrogramTransformer, HeadTokensAggregator, MLPHead

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
    def build_from_default_config(cls, is_teacher: bool = True) -> "MusicTaggingTransformer":
        # PositionalPatchEmbedding
        hidden_channels = 128
        kernel_size = 3
        pool_kernel_size = None
        pool_stride = None
        num_embedding_layers = 3
        num_embedding_blocks = 2
        insert_cls_token = True
        insert_dist_token = False
        embedding_dropout = 0.1
        max_length = 512
        support_extrapolation = False

        # MusicTaggingTransformerEncoder
        if is_teacher:
            d_model = 256
        else:
            d_model = 64

        dim_feedforward = 4 * d_model
        nhead = 8
        activation = "gelu"
        backbone_dropout = 0.1
        num_backbone_layers = 4
        layer_norm_eps = 1e-5
        batch_first = True
        norm_first = True
        bias = True
        norm = None

        n_bins = 128
        num_classes = 50

        embedding = PositionalPatchEmbedding(
            d_model,
            hidden_channels,
            n_bins,
            kernel_size=kernel_size,
            pool_kernel_size=pool_kernel_size,
            pool_stride=pool_stride,
            num_layers=num_embedding_layers,
            num_blocks=num_embedding_blocks,
            insert_cls_token=insert_cls_token,
            insert_dist_token=insert_dist_token,
            dropout=embedding_dropout,
            max_length=max_length,
            support_extrapolation=support_extrapolation,
        )
        backbone = MusicTaggingTransformerEncoder(
            d_model,
            nhead,
            num_layers=num_backbone_layers,
            dim_feedforward=dim_feedforward,
            dropout=backbone_dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            norm=norm,
        )
        aggregator = HeadTokensAggregator(
            insert_cls_token=insert_cls_token,
            insert_dist_token=insert_dist_token,
        )
        head = MLPHead(d_model, num_classes)

        model = cls(embedding, backbone, aggregator=aggregator, head=head)

        return model

    @classmethod
    def build_from_pretrained(cls) -> "MusicTaggingTransformer":
        pass
