from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FastSpeechDurationPredictor", "DurationPredictor"]


class FastSpeechDurationPredictor(nn.Module):
    """Duration predictor of FastSpeech composed of stacked ConvBlock.

    Args:
        num_features (list): Number of features.
        kernel_size (int): Kernel size in convolutions.
        dropout (float): Drop out rate. Default: ``1e-1``.
        stop_gradient (bool): Whether to stop gradient of input feature.
        batch_first (bool): If ``True``, first dimension of input is treated as batch index.
        channels_last (bool): If ``True``, last dimension of input is treated as channel index.

    """

    def __init__(
        self,
        num_features: List[int],
        kernel_size: int = 3,
        dropout: float = 1e-1,
        stop_gradient: bool = True,
        batch_first: bool = False,
        channels_last: bool = True,
    ) -> None:
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
        self.batch_first, self.channels_last = batch_first, channels_last

    def forward(self, input: torch.Tensor, padding_mask: torch.BoolTensor = None) -> torch.Tensor:
        """Forward pass of DurationPredictor.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, length, num_features)
                if ``batch_first=True``, otherwise (length, batch_size, num_features).
            padding_mask (torch.BoolTensor): Padding mask of shape (length,)
                or (batch_size, length).

        Returns:
            torch.Tensor: Estimated log duration of shape (batch_size, length)
                if ``batch_first=True``, otherwise (length, batch_size).

        """
        batch_first, channels_last = self.batch_first, self.channels_last
        stop_gradient = self.stop_gradient

        if stop_gradient:
            x = input.detach()
        else:
            x = input

        # permute axes to (batch_size, num_features, length)
        if batch_first:
            if channels_last:
                x = x.permute(0, 2, 1)
        else:
            if channels_last:
                x = x.permute(1, 2, 0)
            else:
                raise NotImplementedError(
                    "Condition of batch_first = False and channels_last = False is not supported."
                )

        for layer_idx in range(self.num_layers):
            x = self.backbone[layer_idx](x, padding_mask=padding_mask)

        x = x.permute(0, 2, 1)
        x = self.fc_layer(x)
        log_duration = x.squeeze(dim=-1)

        if padding_mask is not None:
            log_duration = log_duration.masked_fill(padding_mask, -float("inf"))

        if not batch_first:
            log_duration = log_duration.permute(1, 0).contiguous()

        return log_duration


class DurationPredictor(FastSpeechDurationPredictor):
    """
    Wrapper class of FastSpeechDurationPredictor.
    """


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size should be odd."

        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
        )
        self.activation1d = nn.ReLU()
        self.norm1d = nn.LayerNorm(out_channels)
        self.dropout1d = nn.Dropout(p=dropout)

        self.kernel_size = kernel_size

    def forward(
        self, input: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
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

        x = self._masked_fill(input, padding_mask=padding_mask)
        x = F.pad(x, (padding_left, padding_right))
        x = self.conv1d(x)
        x = self._masked_fill(x, padding_mask=padding_mask)
        x = self.activation1d(x)
        x = x.permute(0, 2, 1)
        x = self.norm1d(x)
        x = x.permute(0, 2, 1)
        x = self._masked_fill(x, padding_mask=padding_mask)
        output = self.dropout1d(x)

        return output

    @staticmethod
    def _masked_fill(
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Apply padding mask if given.

        Args:
            input (torch.Tensor): Tensor of shape (batch_size, num_features, length).
            padding_mask (torch.BoolTensor): Padding mask of shape (batch_size, length).

        Returns:
            torch.Tensor: Output tensor of same shape as input.

        """
        if padding_mask is None:
            output = input
        else:
            output = input.masked_fill(padding_mask.unsqueeze(dim=-2), 0)

        return output
