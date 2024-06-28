import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Segment1d", "OverlapAdd1d"]


class Segment1d(nn.Module):
    """Segmentation.

    Input tensor is 3-D (audio-like), but output tensor is 4-D (image-like).
    """

    def __init__(self, chunk_size: int, hop_size: int) -> None:
        super().__init__()

        self.chunk_size = chunk_size
        self.hop_size = hop_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of Segment1d.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, num_features, num_frames).

        Returns:
            torch.Tensor: Output feature of shape
                (batch_size, num_features, inter_length, chunk_size),
                where inter_length = (num_frames-chunk_size) // hop_size + 1.

        """
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, num_features, num_frames = input.size()

        input = input.view(batch_size, num_features, num_frames, 1)
        x = F.unfold(input, kernel_size=(chunk_size, 1), stride=(hop_size, 1))
        x = x.view(batch_size, num_features, chunk_size, -1)
        output = x.permute(0, 1, 3, 2).contiguous()

        return output

    def extra_repr(self) -> str:
        s = "chunk_size={chunk_size}, hop_size={hop_size}".format(
            chunk_size=self.chunk_size, hop_size=self.hop_size
        )
        return s


class OverlapAdd1d(nn.Module):
    """Overlap-add operation.

    Input tensor is 4-D (image-like), but output tensor is 3-D (audio-like).
    """

    def __init__(self, chunk_size: int, hop_size: int) -> None:
        super().__init__()

        self.chunk_size = chunk_size
        self.hop_size = hop_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of OverlapAdd1d.

        Args:
            input (torch.Tensor): Input feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, num_features, num_frames).

        """
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, num_features, inter_length, chunk_size = input.size()
        num_frames = (inter_length - 1) * hop_size + chunk_size

        x = input.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, num_features * chunk_size, inter_length)
        output = F.fold(
            x, kernel_size=(chunk_size, 1), stride=(hop_size, 1), output_size=(num_frames, 1)
        )
        output = output.squeeze(dim=3)

        return output

    def extra_repr(self) -> str:
        s = "chunk_size={chunk_size}, hop_size={hop_size}".format(
            chunk_size=self.chunk_size, hop_size=self.hop_size
        )
        return s
