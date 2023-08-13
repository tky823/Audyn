from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FastSpeechDurationPredictor"]


class FastSpeechDurationPredictor(nn.Module):
    def __init__(
        self,
        num_features: List[int],
        kernel_size: int = 3,
        dropout: float = 1e-1,
        stop_gradient: bool = True,
        batch_first: bool = False,
    ):
        super().__init__()

        backbone = []
        self.num_layers = len(num_features) - 1

        for layer_idx in range(self.num_layers):
            backbone.append(
                ConvBlock(
                    num_features[layer_idx],
                    num_features[layer_idx + 1],
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        self.backbone = nn.ModuleList(backbone)
        self.fc_layer = nn.Linear(num_features[-1], 1)

        self.stop_gradient = stop_gradient
        self.batch_first = batch_first

    def forward(self, input: torch.Tensor, padding_mask: torch.BoolTensor = None) -> torch.Tensor:
        """Forward pass of DurationPredictor.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, length, num_features)
                if ``batch_first=True``, otherwise (length, batch_size, num_features).
            padding_mask (torch.BoolTensor): Padding mask of shape (length,)
                or (batch_size, length).

        Returns:
            torch.Tensor: Estimated log duration of shape (batch_size, length).

        """
        batch_first = self.batch_first
        stop_gradient = self.stop_gradient

        if stop_gradient:
            x = input.detach()
        else:
            x = input

        if batch_first:
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(1, 2, 0)

        for layer_idx in range(self.num_layers):
            x = self.backbone[layer_idx](x, padding_mask=padding_mask)

        x = x.permute(0, 2, 1)
        x = self.fc_layer(x)
        log_duration = x.squeeze(dim=-1)

        if padding_mask is not None:
            log_duration = log_duration.masked_fill(padding_mask, -float("inf"))

        return log_duration


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size should be odd."

        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=False
        )
        self.norm1d = nn.LayerNorm(out_channels)
        self.activation1d = nn.ReLU()
        self.dropout1d = nn.Dropout(p=dropout)

        self.kernel_size = kernel_size

    def forward(self, input: torch.Tensor, padding_mask: torch.BoolTensor = None) -> torch.Tensor:
        """Forward pass of ConvBlock.

        Args:
            input (torch.Tensor): Input feature with shape of (batch_size, num_features, length).
            padding_mask (torch.BoolTensor): Padding mask of shape (length,)
                or (batch_size, length).

        Returns:
            torch.Tensor: Output feature with shape of (batch_size, num_features, length).

        """
        kernel_size = self.kernel_size

        padding_left = (kernel_size - 1) // 2
        padding_right = kernel_size - 1 - padding_left
        x = F.pad(input, (padding_left, padding_right))
        x = self.conv1d(x)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(dim=-2), 0)

        x = x.permute(0, 2, 1)
        x = self.norm1d(x)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(dim=-1), 0)

        x = self.activation1d(x)
        x = self.dropout1d(x)
        output = x.permute(0, 2, 1)

        return output
