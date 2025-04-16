import warnings

import torch
from torchaudio.functional.functional import (
    _create_triangular_filterbank,
    _hz_to_mel,
    _mel_to_hz,
)

__all__ = [
    "melscale_fbanks",
]


def melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: str | None = None,
    mel_scale: str = "htk",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # ported from https://github.com/pytorch/audio/blob/bccaa454a54c3c648697cc2f46a4fb0500b1f01b/src/torchaudio/functional/functional.py#L521-L590  # noqa: E501
    r"""Create a frequency bin conversion matrix for Mel-spectrogram.

    .. note::

        Unlike ``torchaudio.functional.melscale_fbanks``, ``dtype`` is supported
        for high precision.

    """

    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs, dtype=dtype)

    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2, dtype=dtype)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)

    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if norm is not None and norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb
