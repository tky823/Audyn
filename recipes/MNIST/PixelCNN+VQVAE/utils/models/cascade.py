from typing import Any

import torch
import torch.nn as nn
from utils.models.pixelcnn import PixelCNN

from audyn.models.vqvae import VQVAE


class PixelCNNVQVAE(nn.Module):
    """Cascade of PixelCNN and VQVAE"""

    def __init__(self, pixelcnn: PixelCNN, vqvae: VQVAE) -> None:
        super().__init__()

        self.pixelcnn = pixelcnn
        self.vqvae = vqvae

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("forward pass is not supported.")

    @torch.no_grad()
    def inference(
        self, initial_state: torch.Tensor, height: int = 1, width: int = 1
    ) -> torch.Tensor:
        """Forward pass of PixelCNN + VQVAE.

        Args:
            initial_state (torch.Tensor): (batch_size, 1, 1).
            height (int): Height of output tensor.
            width (int): Width of output tensor.

        Returns:
            torch.Tensor: (batch_size, out_channels, height, width).

        """
        quantized = self.pixelcnn.inference(initial_state, height=height, width=width)
        output = self.vqvae.inference(quantized)

        return output
