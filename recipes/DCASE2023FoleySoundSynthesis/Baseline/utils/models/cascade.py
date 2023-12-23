from typing import Any

import torch
import torch.nn as nn
from utils.models.pixelsnail import PixelSNAIL

from audyn.models.hifigan import Generator
from audyn.models.vqvae import VQVAE


class BaselineModel(nn.Module):
    """Cascade of PixelSNAIL, VQVAE, and HiFi-GAN.

    Args:
        pixelsnail: PixelSNAIL.
        vqvae: VQVAE.
        hifigan_generator: Generator of HiFi-GAN.

    """

    def __init__(self, pixelsnail: PixelSNAIL, vqvae: VQVAE, hifigan_generator: Generator) -> None:
        super().__init__()

        self.pixelsnail = pixelsnail
        self.vqvae = vqvae
        self.hifigan_generator = hifigan_generator

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("forward pass is not supported.")

    @torch.no_grad()
    def inference(
        self, initial_state: torch.Tensor, height: int = 1, width: int = 1
    ) -> torch.Tensor:
        """Forward pass of PixelSNAIL + VQVAE + HiFi-GAN.

        Args:
            initial_state (torch.Tensor): (batch_size, 1, 1).
            height (int): Height of latent codes in VQVAE.
            width (int): Width of latent codes in VQVAE.

        Returns:
            torch.Tensor: (batch_size, out_channels, length).

        """
        quantized = self._inference(self.pixelsnail, initial_state, height=height, width=width)
        melspectrogram = self._inference(self.vqvae, quantized)
        output = self._inference(self.hifigan_generator, melspectrogram)

        return output

    @staticmethod
    def _inference(module: nn.Module, *args, **kwargs) -> torch.Tensor:
        if hasattr(module, "inference"):
            return module.inference(*args, **kwargs)
        else:
            return module(*args, **kwargs)
