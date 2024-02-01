from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from utils.models.transformer import _Transformer


class CLAP(nn.Module):
    def __init__(self, text_tower: "ModalTower", audio_tower: "ModalTower") -> None:
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


class ModalTower(nn.Module):
    def __init__(self, backbone: nn.Module, out_proj: nn.Linear) -> None:
        super().__init__()

        if isinstance(backbone, _Transformer):
            pass
        else:
            raise ValueError(f"{type(backbone)} is not supported as backbone.")

        self.backbone = backbone
        self.out_proj = out_proj

    def forward(
        self,
        input: Union[torch.LongTensor, torch.Tensor],
        length: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        x = self.backbone(input, length=length)
        output = self.out_proj(x)

        return output
