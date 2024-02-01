from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from audyn.modules.positional_encoding import AbsolutePositionalEncoding

available_aggregations = ["cls", "pool", "none"]


class _Transformer(nn.Module):
    cls_embedding: nn.Parameter
    positional_encoding: AbsolutePositionalEncoding
    backbone: nn.TransformerEncoder

    embedding_dim: int
    batch_first: bool
    aggregation: str

    def transformer_forward(
        self,
        input: torch.LongTensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_first = self.batch_first
        cls_embedding = self.cls_embedding

        factory_kwargs = {"device": input.device}

        if batch_first:
            batch_size, max_length, _ = input.size()
        else:
            max_length, batch_size, _ = input.size()

        if length is None:
            length = torch.full((batch_size,), fill_value=max_length, **factory_kwargs)

        padding_mask = torch.arange(max_length, **factory_kwargs) >= length.unsqueeze(dim=-1)

        if batch_first:
            expanded_padding_mask = padding_mask
        else:
            expanded_padding_mask = padding_mask.permute(1, 0)

        expanded_padding_mask = expanded_padding_mask.unsqueeze(dim=-1)

        x = input.masked_fill(expanded_padding_mask, 0)
        x = self.positional_encoding(x)
        x = x.masked_fill(expanded_padding_mask, 0)
        cls_embedding = cls_embedding.view(1, 1, -1)

        if batch_first:
            cls_embedding = cls_embedding.expand(batch_size, 1, -1)
            x = torch.cat([cls_embedding, x], dim=1)
        else:
            cls_embedding = cls_embedding.expand(1, batch_size, -1)
            x = torch.cat([cls_embedding, x], dim=0)

        padding_mask = F.pad(padding_mask, (1, 0), value=False)

        x = self.backbone(x, src_key_padding_mask=padding_mask)

        output = self.aggregate(x, length=length)

        return output

    def aggregate(
        self,
        input: torch.Tensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        aggregation = self.aggregation
        batch_first = self.batch_first

        factory_kwargs = {"device": input.device}

        # NOTE: max_length includes cls token.
        if batch_first:
            batch_size, max_length, _ = input.size()
        else:
            max_length, batch_size, _ = input.size(0)

        if length is None:
            length = torch.full((batch_size,), fill_value=max_length, **factory_kwargs)

        if batch_first:
            cls_token, output = torch.split(input, [1, max_length - 1], dim=1)
        else:
            cls_token, output = torch.split(input, [1, max_length - 1], dim=0)

        if aggregation == "cls":
            if batch_first:
                output = cls_token.squeeze(dim=1)
            else:
                output = cls_token.squeeze(dim=0)
        elif aggregation == "pool":
            if batch_first:
                output = output.sum(dim=1) / length.unsqueeze(dim=1)
            else:
                output = output.sum(dim=0) / length.unsqueeze(dim=0)
        elif aggregation == "none":
            output = input
        else:
            raise ValueError(f"{aggregation} is not supported as aggregation.")

        return output

    def _reset_parameters(self) -> None:
        self.cls_embedding.data.normal_()


class TextTransformer(_Transformer):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        nhead: int,
        num_layers: int = 6,
        batch_first: bool = True,
        aggregation: str = "cls",
    ) -> None:
        super().__init__()

        assert aggregation in available_aggregations

        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim,
            nhead,
            dim_feedforward=embedding_dim,
            batch_first=batch_first,
        )

        self.cls_embedding = nn.Parameter(torch.empty((embedding_dim,)))
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = AbsolutePositionalEncoding(batch_first=batch_first)
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.embedding_dim = embedding_dim
        self.batch_first = batch_first
        self.aggregation = aggregation

        self._reset_parameters()

    def forward(
        self,
        input: torch.LongTensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        x = self.word_embedding(input)
        output = self.transformer_forward(x, length=length)

        return output


class AudioTransformer(_Transformer):
    def __init__(
        self,
        embedding_dim: int,
        nhead: int,
        num_layers: int = 6,
        batch_first: bool = True,
        channels_last: bool = True,
        aggregation: str = "cls",
    ) -> None:
        super().__init__()

        assert aggregation in available_aggregations

        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim,
            nhead,
            dim_feedforward=embedding_dim,
            batch_first=batch_first,
        )

        self.cls_embedding = nn.Parameter(torch.empty((embedding_dim,)))
        self.positional_encoding = AbsolutePositionalEncoding(batch_first=batch_first)
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.embedding_dim = embedding_dim
        self.channels_last = channels_last
        self.batch_first = batch_first
        self.aggregation = aggregation

        if (not batch_first) and (not channels_last):
            raise ValueError("Either of batch_first or channels_last should be True.")

        self._reset_parameters()

    def forward(
        self,
        input: torch.LongTensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if not self.channels_last:
            input = input.transpose(-2, -1).contiguous()

        output = self.transformer_forward(input, length=length)

        return output
