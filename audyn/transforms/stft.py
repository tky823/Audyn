import torch
import torch.nn as nn

__all__ = [
    "ShortTimeFourierTransform",
    "InverseShortTimeFourierTransform",
    "STFT",
    "ISTFT",
]


class ShortTimeFourierTransform(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        if "window" in kwargs:
            window = kwargs.pop("window")
        else:
            window = None

        self.kwargs = kwargs
        self.register_buffer("window", window)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        window = self.window

        *batch_shape, length = input.size()

        x = input.contiguous()
        x = x.view(-1, length)
        x = torch.stft(x, window=window, **self.kwargs)
        *_, n_bins, n_frames = x.size()
        x = x.contiguous()
        output = x.view(*batch_shape, n_bins, n_frames)

        return output


class InverseShortTimeFourierTransform(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        if "window" in kwargs:
            window = kwargs.pop("window")
        else:
            window = None

        self.kwargs = kwargs
        self.register_buffer("window", window)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        window = self.window

        *batch_shape, length = input.size()

        x = input.contiguous()
        x = x.view(-1, length)
        x = torch.istft(x, window=window, **self.kwargs)
        *_, n_bins, n_frames = x.size()
        x = x.contiguous()
        output = x.view(*batch_shape, n_bins, n_frames)

        return output


class STFT(ShortTimeFourierTransform):
    """Alias of ShortTimeFourierTransform."""


class ISTFT(InverseShortTimeFourierTransform):
    """Alias of InverseShortTimeFourierTransform."""
