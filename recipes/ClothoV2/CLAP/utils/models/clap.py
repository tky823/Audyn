import importlib
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from utils.models.aggregator import Aggregator
from utils.models.transformer import (
    AudioTransformerBackbone,
    AudioTransformerMaskedPatchModel,
    AudioTransformerMaskedPatchModelBackbone,
    TextTransformerBackbone,
    TextTransformerMaskedLanguageModel,
    TextTransformerMaskedLanguageModelBackbone,
    TransformerBackbone,
)

from audyn.utils import instantiate_model


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

    def no_aggregation_forward(
        self,
        text: torch.LongTensor,
        audio: torch.Tensor,
        text_length: Optional[torch.LongTensor] = None,
        audio_length: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_output = self.text_tower.no_aggregation_forward(text, length=text_length)
        audio_output = self.audio_tower.no_aggregation_forward(audio, length=audio_length)

        return text_output, audio_output


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

    @classmethod
    def build_from_pretrained(
        cls,
        path: str,
        aggregation: str,
        out_channels: Optional[int] = None,
    ) -> "ModalTransformerTower":
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        resolved_config = state_dict["resolved_config"]
        resolved_config = OmegaConf.create(resolved_config)
        model_state_dict = state_dict["model"]

        mod_name, var_name = resolved_config.model._target_.rsplit(".", maxsplit=1)
        saved_cls = getattr(importlib.import_module(mod_name), var_name)

        if saved_cls is ModalTransformerTower:
            model = instantiate_model(resolved_config.model)
        elif saved_cls is TextTransformerMaskedLanguageModel:
            required_keys = {
                "vocab_size",
                "embedding_dim",
                "nhead",
            }
            optional_keys = {
                "num_layers",
                "batch_first",
            }
            backbone_config = {"_target_": "utils.models.transformer.TextTransformerBackbone"}

            for key in required_keys:
                backbone_config[key] = resolved_config.model.backbone[key]

            for key in optional_keys:
                if key in resolved_config.model.backbone:
                    backbone_config[key] = resolved_config.model.backbone[key]

            backbone_config = OmegaConf.create(backbone_config)
            model: TextTransformerBackbone = instantiate_model(backbone_config)
        elif saved_cls is AudioTransformerMaskedPatchModel:
            required_keys = {
                "in_channels",
                "embedding_dim",
                "frames_per_patch",
                "nhead",
            }
            optional_keys = {
                "num_layers",
                "batch_first",
                "channels_last",
            }
            backbone_config = {"_target_": "utils.models.transformer.AudioTransformerBackbone"}

            for key in required_keys:
                backbone_config[key] = resolved_config.model.backbone[key]

            for key in optional_keys:
                if key in resolved_config.model.backbone:
                    backbone_config[key] = resolved_config.model.backbone[key]

            backbone_config = OmegaConf.create(backbone_config)
            model: AudioTransformerBackbone = instantiate_model(backbone_config)
        else:
            raise ValueError(f"Unsupported {saved_cls} is detected.")

        if isinstance(model, (TextTransformerBackbone, AudioTransformerBackbone)):
            backbone: Union[TextTransformerBackbone, AudioTransformerBackbone] = model

            batch_first = backbone.batch_first
            aggregator = Aggregator(batch_first=batch_first, aggregation=aggregation)

            if out_channels is None:
                raise ValueError("out_channels is required.")
            else:
                prefix = "backbone."
                backbone_state_dict = OrderedDict()

                for key in list(model_state_dict.keys()):
                    if key.startswith(prefix):
                        backbone_key = key[len(prefix) :]
                        backbone_state_dict[backbone_key] = model_state_dict[key]
                    else:
                        model_state_dict.pop(key)

                backbone.load_state_dict(backbone_state_dict)
                embedding_dim = backbone.embedding_dim
                out_proj = nn.Linear(embedding_dim, out_channels)

            model = cls(backbone, aggregator, out_proj)
        else:
            model: ModalTransformerTower
            model.load_state_dict(model_state_dict)

        return model


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
