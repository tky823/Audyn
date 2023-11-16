import torch
import torch.nn as nn


class NormalizeIndex(nn.Module):
    """Normalize quantization index by max value.

    Args:
        max (int): Max value of quantization index in image.

    """

    def __init__(self, max: int) -> None:
        super().__init__()

        self.max = max

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of NormalizeImage.

        Args:
            input (torch.LongTensor): Quantized indices in [0, max - 1].

        Returns:
            torch.Tensor: Normalized indices in quantization.

        """
        output = input / self.max

        return output
