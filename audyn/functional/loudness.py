import math
from typing import Union

import torch
import torch.nn.functional as F
import torchaudio.functional as aF

__all__ = [
    "compute_loudness",
    "normalize_by_loudness",
]


def compute_loudness(
    waveform: torch.Tensor, sample_rate: int, block_size: float = 0.4
) -> torch.Tensor:
    """Compute loudness of waveform based on ITU-R BS.1770-4.

    Args:
        waveform (torch.Tensor): Waveform of shape (*, num_channels, timesteps).
            ``num_channels`` can be 1 (monaural) or 2 (stereo).
        sample_rate (int): Sample rate of waveform.
        block_size (float): Block size in seconds. Default: ``0.4``.

    Returns:
        torch.Tensor: Loudness in dB LUFS of shape (*,).

    """
    assert waveform.dim() >= 2, "Waveform must have at least 2 dimensions."
    assert waveform.size(-2) <= 2, "Only monaural and stero waveform are supported.."

    waveform = _apply_peaking_filter(waveform, sample_rate=sample_rate)
    waveform = _apply_highpass_filter(waveform, sample_rate=sample_rate)

    hop_length = int(0.25 * block_size * sample_rate)
    win_length = int(block_size * sample_rate)
    num_blocks = round((waveform.size(-1) - win_length) / hop_length) + 1

    *batch_shape, num_channels, timesteps = waveform.size()
    waveform = waveform.view(-1, 1, 1, timesteps)
    padding = win_length + hop_length * (num_blocks - 1) - waveform.size(-1)
    waveform = F.pad(waveform, (0, padding))

    _waveform = F.unfold(waveform, kernel_size=(1, win_length), stride=(1, hop_length))
    power = torch.mean(_waveform**2, dim=-2)
    power = power.view(*batch_shape, num_channels, -1)
    _power = power.sum(dim=-2)
    loudness = -0.691 + 10.0 * torch.log10(_power)

    ath = -70
    non_padding_abolute_mask = loudness >= ath
    padding_abolute_mask = torch.logical_not(non_padding_abolute_mask)
    num_gating_blocks = non_padding_abolute_mask.sum(dim=-1)
    num_gating_blocks = num_gating_blocks.view(*batch_shape, 1)
    padding_abolute_mask = padding_abolute_mask.view(*batch_shape, 1, -1)
    absolute_gating_power = power.masked_fill(padding_abolute_mask, 0)
    absolute_gating_power = absolute_gating_power.sum(dim=-1) / num_gating_blocks
    absolute_gating_power = absolute_gating_power.sum(dim=-1)

    # relative
    rth = -0.691 + 10.0 * torch.log10(absolute_gating_power) - 10
    non_padding_abolute_mask = loudness > ath
    non_padding_relative_mask = loudness > rth.unsqueeze(dim=-1)
    non_padding_mask = non_padding_abolute_mask & non_padding_relative_mask
    padding_mask = torch.logical_not(non_padding_mask)
    num_gating_blocks = non_padding_mask.sum(dim=-1)
    num_gating_blocks = num_gating_blocks.view(*batch_shape, 1)
    padding_mask = padding_mask.view(*batch_shape, 1, -1)
    gating_power = power.masked_fill(padding_mask, 0)
    gating_power = gating_power.sum(dim=-1) / num_gating_blocks
    gating_power = torch.nan_to_num(gating_power)
    gating_power = gating_power.sum(dim=-1)

    loudness = -0.691 + 10 * torch.log10(gating_power)

    return loudness


def normalize_by_loudness(
    waveform: torch.Tensor,
    sample_rate: int,
    loudness: Union[float, torch.Tensor],
    block_size: float = 0.4,
) -> torch.Tensor:
    """Normalize waveform by target loudness level.

    Args:
        waveform (torch.Tensor): Waveform of shape (*, num_channels, timesteps).
            ``num_channels`` can be 1 (monaural) or 2 (stereo).
        sample_rate (int): Sample rate of waveform.
        loudness (float or torch.Tensor): Target loudness in Loudness Units Full Scale (LUFS).
            Shape of (*,) is also supported if ``loudness`` is ``torch.Tensor``.
        block_size (float): Block size in seconds to compute loudness. Default: ``0.4``.

    Returns:
        torch.Tensor: Normalized waveform of shape (*, num_channels, timesteps).

    """
    assert waveform.dim() >= 2, "Waveform must have at least 2 dimensions."

    *batch_shape, _, _ = waveform.size()
    _loudness = compute_loudness(waveform, sample_rate, block_size=block_size)

    _loudness = _loudness.view(*batch_shape, 1, 1)

    if isinstance(loudness, torch.Tensor):
        loudness = loudness.view(*batch_shape, 1, 1)

    gain = 10 ** ((loudness - _loudness) / 20)
    waveform = gain * waveform

    return waveform


def _apply_peaking_filter(
    waveform: torch.Tensor,
    sample_rate: int,
    gain: float = 4.0,
    freq: float = 1500.0,
    q_factor: float = 1 / math.sqrt(2),
) -> torch.Tensor:
    normalized_freq = 2.0 * math.pi * (freq / sample_rate)
    amplitude = 10 ** (gain / 40.0)

    if isinstance(normalized_freq, torch.Tensor):
        _sin = torch.sin(normalized_freq)
        _cos = torch.cos(normalized_freq)
    else:
        _sin = math.sin(normalized_freq)
        _cos = math.cos(normalized_freq)

    if isinstance(amplitude, torch.Tensor):
        _sqrt = torch.sqrt(amplitude)
    else:
        _sqrt = math.sqrt(amplitude)

    alpha = _sin / (2.0 * q_factor)

    b0 = amplitude * ((amplitude + 1) + (amplitude - 1) * _cos + 2 * _sqrt * alpha)
    b1 = -2 * amplitude * ((amplitude - 1) + (amplitude + 1) * _cos)
    b2 = amplitude * ((amplitude + 1) + (amplitude - 1) * _cos - 2 * _sqrt * alpha)
    a0 = (amplitude + 1) - (amplitude - 1) * _cos + 2 * _sqrt * alpha
    a1 = 2 * ((amplitude - 1) - (amplitude + 1) * _cos)
    a2 = (amplitude + 1) - (amplitude - 1) * _cos - 2 * _sqrt * alpha

    waveform = aF.biquad(waveform, b0 / a0, b1 / a0, b2 / a0, a0 / a0, a1 / a0, a2 / a0)

    return waveform


def _apply_highpass_filter(
    waveform: torch.Tensor,
    sample_rate: int,
    freq: float = 38.0,
    q_factor: float = 0.5,
) -> torch.Tensor:
    normalized_freq = 2.0 * math.pi * (freq / sample_rate)

    if isinstance(normalized_freq, torch.Tensor):
        _sin = torch.sin(normalized_freq)
        _cos = torch.cos(normalized_freq)
    else:
        _sin = math.sin(normalized_freq)
        _cos = math.cos(normalized_freq)

    alpha = _sin / (2.0 * q_factor)

    b0 = (1 + _cos) / 2
    b1 = -(1 + _cos)
    b2 = (1 + _cos) / 2
    a0 = 1 + alpha
    a1 = -2 * _cos
    a2 = 1 - alpha

    waveform = aF.biquad(waveform, b0 / a0, b1 / a0, b2 / a0, a0 / a0, a1 / a0, a2 / a0)

    return waveform
