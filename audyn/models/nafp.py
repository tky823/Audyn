from typing import Tuple

import torch
import torch.nn as nn

__all__ = [
    "NeuralAudioFingerprinter",
    "ContrastiveNeuralAudioFingerprinter",
]


class NeuralAudioFingerprinter(nn.Module):
    """Neural Audio Fingerprinter.

    Args:
        backbone (nn.Module): Backbone for feature extraction.
        projection (nn.Module): Projection for embedding.

    """

    def __init__(self, backbone: nn.Module, projection: nn.Module) -> None:
        super().__init__()

        self.backbone = backbone
        self.projection = projection

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of NeuralAudioFingerprinter.

        Args:
            input (torch.Tensor): Spectrogram-like feature of shape (*, n_bins, n_frames).

        Returns:
            torch.Tensor: Fingerprint of shape (*, embedding_dim).

        """
        x = self.backbone(input)
        output = self.projection(x)

        return output


class ContrastiveNeuralAudioFingerprinter(NeuralAudioFingerprinter):
    """Wrapper class of NeuralAudioFingerprinter for contrastive learning."""

    def forward(
        self, input: torch.Tensor, other: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of ContrastiveNeuralAudioFingerprinter.

        Args:
            input (torch.Tensor): Spectrogram-like feature of shape (*, n_bins, n_frames).
            other (torch.Tensor): Augmented feature of shape (*, n_bins, n_frames).

        Returns:
            tuple: Tuple of torch.Tensor containing:
                - torch.Tensor: Fingerprint of shape (*, embedding_dim).
                - torch.Tensor: Fingerprint for augmented feature of shape (*, embedding_dim).

        """
        x = torch.cat([input, other], dim=0)
        x = super().forward(x)
        split_size_or_sections = x.size(0) // 2
        output_one, output_other = torch.split(x, split_size_or_sections, dim=0)

        return output_one, output_other
