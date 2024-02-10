from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from utils.models.aggregator import Aggregator
from utils.models.transformer import (
    AudioTransformerBackbone,
    AudioTransformerMaskedPatchModelBackbone,
    TextTransformerBackbone,
    TextTransformerMaskedLanguageModelBackbone,
    TransformerBackbone,
)


class CLAP(nn.Module):
    def __init__(
        self,
        text_tower: "TextTransformerTower",
        audio_tower: "AudioTransformerTower",
    ) -> None:
        super().__init__()

        self.text_tower = text_tower
        self.audio_tower = audio_tower

    def forward(
        self,
        text: torch.LongTensor,
        audio: torch.Tensor,
        text_length: Optional[torch.LongTensor] = None,
        audio_length: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_embedding = self.text_tower(text, length=text_length)
        audio_embedding = self.audio_tower(audio, length=audio_length)

        return text_embedding, audio_embedding


class ModalTransformerTower(nn.Module):
    """Base class of TransformerTower."""

    def __init__(
        self,
        backbone: TransformerBackbone,
        aggregator: Aggregator,
        out_proj: nn.Linear,
    ) -> None:
        super().__init__()

        assert isinstance(backbone, TransformerBackbone)

        self.backbone = backbone
        self.aggregator = aggregator
        self.out_proj = out_proj

    def forward(
        self,
        input: Union[torch.Tensor, torch.LongTensor],
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        x = self.backbone(input, length=length)
        x = self.aggregator(x, length=length)
        output = self.out_proj(x)

        return output

    def no_aggregation_forward(
        self,
        input: torch.Tensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_first = self.backbone.batch_first

        x = self.backbone(input, length=length)

        if batch_first:
            dim = 1
        else:
            dim = 0

        max_length = x.size(dim) - 1
        _, x = torch.split(x, [1, max_length], dim=dim)
        output = self.out_proj(x)

        return output


class TextTransformerTower(ModalTransformerTower):
    def __init__(
        self,
        backbone: TextTransformerBackbone,
        aggregator: Aggregator,
        out_proj: nn.Linear,
    ) -> None:
        super().__init__(backbone, aggregator, out_proj)

        assert not isinstance(backbone, TextTransformerMaskedLanguageModelBackbone)


class AudioTransformerTower(ModalTransformerTower):
    def __init__(
        self,
        backbone: AudioTransformerBackbone,
        aggregator: Aggregator,
        out_proj: nn.Linear,
    ) -> None:
        super().__init__(backbone, aggregator, out_proj)

        assert not isinstance(backbone, AudioTransformerMaskedPatchModelBackbone)
