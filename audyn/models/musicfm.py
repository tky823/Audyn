import warnings
from typing import Optional

import torch
import torch.nn as nn


class MusicFM(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        backbone: nn.Module,
        aggregator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.backbone = backbone
        self.aggregator = aggregator
        self.head = head

        if self.aggregator is None and self.head is not None:
            warnings.warn(
                "Head is given, but aggregator is not given, "
                "which may lead to unexpected behavior.",
                UserWarning,
                stacklevel=2,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            x = input.unsqueeze(dim=-3)
        elif input.dim() == 4:
            x = input
        else:
            raise ValueError("Only 3D and 4D inputs are supported.")

        x = self.embedding(x)
        output = self.backbone(x)

        if self.aggregator is not None:
            output = self.aggregator(output)

        if self.head is not None:
            output = self.head(output)

        return output
