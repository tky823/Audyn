import torch
import torch.nn as nn


class MusicFM(nn.Module):
    def __init__(self, embedding: nn.Module, backbone: nn.Module) -> None:
        super().__init__()

        self.embedding = embedding
        self.backbone = backbone

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            x = input.unsqueeze(dim=-3)
        elif input.dim() == 4:
            x = input
        else:
            raise ValueError("Only 3D and 4D inputs are supported.")

        x = self.embedding(x)
        output = self.backbone(x)

        return output
