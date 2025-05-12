from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from audyn.utils.alignment import expand_by_duration


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear = nn.Linear(1, 1, bias=False)
        self.linear.weight.data.fill_(1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.linear(input)

        return output

    @torch.no_grad()
    def inference(self, input: torch.Tensor) -> torch.Tensor:
        output = self.linear(input)

        return output


class DummyAutoregressiveFeatToWave(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1d = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        output = self.conv1d(waveform)

        return output

    @torch.no_grad()
    def inference(self, initial_state: torch.Tensor, max_length: int) -> torch.Tensor:
        if initial_state.dim() == 2:
            initial_state = initial_state.unsqueeze(dim=-1)

        state = initial_state
        output = []

        for _ in range(max_length):
            state = self.conv1d(state)
            output.append(state)

        output = torch.stack(output, dim=0)

        return output


class DummyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1d = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        output = self.conv1d(noise)

        return output


class DummyDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1d = nn.Conv1d(1, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(waveform)
        x = x.mean(dim=(1, 2))
        output = self.sigmoid(x)

        return output


class DummyTextToFeat(nn.Module):
    def __init__(self, vocab_size: int, num_features: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, num_features)

    def forward(
        self,
        input: torch.LongTensor,
        duration: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass of DummyTextToFeat.

        Args:
            input (torch.LongTensor): Text tokens of shape (batch_size, src_length).
            duration (torch.LongTensor, optional): Text durations of shape
                (batch_size, src_length).
            max_length (int, optional): Max length of output.

        Returns:
            torch.Tensor: Estimated feature of shape (batch_size, tgt_length, num_features).

        """
        if duration is None:
            duration = 2 * torch.ones_like(input, dtype=torch.long)

        x = self.embedding(input)
        x = expand_by_duration(x, duration)

        _, tgt_length, _ = x.size()

        if max_length is not None and tgt_length > max_length:
            output = F.pad(x, (0, 0, 0, max_length - tgt_length))
        else:
            output = x

        return output

    def inference(self, input: torch.LongTensor, max_length: Optional[int] = None) -> torch.Tensor:
        """Forward pass of DummyTextToFeat.

        Args:
            input (torch.LongTensor): Text tokens of shape (batch_size, src_length).
            max_length (int, optional): Max length of output.

        Returns:
            torch.Tensor: Estimated feature of shape (batch_size, tgt_length, num_features).

        """
        output = self(input, max_length=max_length)

        return output


class DummyFeatToWave(nn.Module):
    def __init__(self, num_features: int, up_scale: int = 2) -> None:
        super().__init__()

        self.kernel_size = 4
        self.up_scale = up_scale

        self.upsample = nn.ConvTranspose1d(
            num_features, 1, kernel_size=self.kernel_size, stride=up_scale
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of DummyFeatToWave.

        Args:
            input (torch.Tensor): Acoustic features of shape
                (batch_size, num_features, length).

        Returns:
            torch.Tensor: Estimated feature of shape (batch_size, 1, up_scale * length).

        """
        kernel_size, up_scale = self.kernel_size, self.up_scale
        padding = kernel_size - up_scale
        padding_left = padding // 2
        padding_right = padding - padding_left

        x = self.upsample(input)
        output = F.pad(x, (-padding_left, -padding_right))

        return output


class DummyCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(input)
        output = x.mean(dim=(1, 2))

        return output


class DummyWaveformProcessor(nn.Module):
    def __init__(self, is_monoral: bool) -> None:
        super().__init__()

        if is_monoral:
            num_channels = 1
        else:
            num_channels = 2

        self.linear = nn.Linear(num_channels, num_channels)

        self.is_monoral = is_monoral

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.is_monoral:
            x = input.unsqueeze(dim=-1)
        else:
            x = input.transpose(-2, -1)

        x = self.linear(x)

        if self.is_monoral:
            output = x.squeeze(dim=-1)
        else:
            output = x.transpose(-2, -1)

        return output
