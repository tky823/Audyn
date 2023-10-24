import torch
import torch.nn as nn


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


class DummyCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(input)
        output = x.mean(dim=(1, 2))

        return output
