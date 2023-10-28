from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from audyn.modules.pixelcnn import HorizontalConv2d, VerticalConv2d


class PixelCNN(nn.Module):
    """PixelCNN for images."""

    def __init__(
        self,
        num_classes: int,
        hidden_channels: int,
        kernel_size: int = 5,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(num_classes, hidden_channels)

        backbone = []

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                capture_center = False
            else:
                capture_center = True

            if layer_idx == num_layers - 1:
                dual_head = False
            else:
                dual_head = True

            block = GatedConv2d(
                hidden_channels,
                kernel_size=kernel_size,
                capture_center=capture_center,
                dual_head=dual_head,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)
        self.post_net = PostNet(hidden_channels, num_classes, 2 * hidden_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of PixelCNN.

        Args:
            input (torch.Tensor): (batch_size, height, width).


        Returns:
            torch.Tensor: (batch_size, num_classes, height, width).

        """
        x = self.embedding(input)
        x = x.permute(0, 3, 1, 2)

        vertical, horizontal = x, x

        for layer in self.backbone:
            vertical, horizontal = layer(vertical, horizontal)

        output = self.post_net(horizontal)

        return output

    @torch.no_grad()
    def inference(
        self,
        initial_state: torch.LongTensor,
        height: int = 1,
        width: int = 1,
    ) -> torch.Tensor:
        """Forward pass of PixelCNN.

        Args:
            initial_state (torch.Tensor): (batch_size, 1, 1).
            height (int): Height of output tensor.
            width (int): Width of output tensor.

        Returns:
            torch.Tensor: (batch_size, height, width).

        """
        # remove redundancy
        batch_size = initial_state.size(0)
        output = F.pad(initial_state, (0, width - 1, 0, height - 1))

        for row_idx in range(height):
            for column_idx in range(width):
                x = F.pad(output, (0, 0, 0, -(height - 1 - row_idx)))
                x = self.forward(x)
                last_output = F.pad(x, (-column_idx, -(width - 1 - column_idx), -row_idx, 0))
                last_output = last_output.view(batch_size, -1)

                # sampling from categorical distribution
                last_output = torch.softmax(last_output, dim=1)
                last_output = torch.distributions.Categorical(last_output).sample()
                output[:, row_idx, column_idx] = last_output

        return output


class GatedConv2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        kernel_size: int,
        capture_center: bool = True,
        dual_head: bool = True,
    ) -> None:
        super().__init__()

        self.vertical_conv2d = VerticalConv2d(
            num_features,
            2 * num_features,
            kernel_size=kernel_size,
        )
        self.horizontal_conv2d = HorizontalConv2d(
            num_features,
            2 * num_features,
            kernel_size=kernel_size,
            capture_center=capture_center,
        )
        self.bridge_conv2d = nn.Conv2d(2 * num_features, 2 * num_features, kernel_size=1, stride=1)
        self.pointwise_conv2d = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1)

        self.capture_center = capture_center
        self.dual_head = dual_head

    def forward(
        self, vertical: torch.Tensor, horizontal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = horizontal

        x_vertical = self.vertical_conv2d(vertical)
        x_horizontal = self.horizontal_conv2d(horizontal)
        x_horizontal = x_horizontal + self.bridge_conv2d(x_vertical)

        if self.dual_head:
            x_tanh, x_sigmoid = torch.chunk(x_vertical, chunks=2, dim=1)
            vertical = torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)
        else:
            vertical = None

        x_tanh, x_sigmoid = torch.chunk(x_horizontal, chunks=2, dim=1)
        x_horizontal = torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)
        x_horizontal = self.pointwise_conv2d(x_horizontal)

        if self.capture_center:
            horizontal = residual + x_horizontal
        else:
            # When capture_center is False (e.g. first layer),
            # causality should be ensured.
            # Thus, residual path cannot be passed to next layer.
            horizontal = x_horizontal

        return vertical, horizontal


class PostNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int) -> None:
        super().__init__()

        self.conv2d_in = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
        )
        self.relu = nn.ReLU()
        self.conv2d_out = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv2d_in(input)
        x = self.relu(x)
        output = self.conv2d_out(x)

        return output
