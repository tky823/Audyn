import math

import pytest
import torch
import torchaudio

from audyn.transforms.cqt import ConstantQTransform, build_temporal_kernel, compute_filter_length


@pytest.mark.parametrize("n_bins", [12, 24])
def test_constant_q_transform(n_bins: int) -> None:
    torch.manual_seed(0)

    waveform = []
    sample_rate = 22050
    # from librosa
    paths = [
        "https://librosa.org/data/audio/sorohanro_-_solo-trumpet-06.ogg",
        "https://librosa.org/data/audio/Kevin_MacLeod_-_Vibe_Ace.ogg",
        "https://librosa.org/data/audio/147793__setuniman__sweet-waltz-0i-22mi.ogg",
    ]
    timesteps = None

    batch_size = len(paths)
    f_min = 440
    bins_per_octave = 12
    original_hop_length = 5

    num_repeats = math.ceil(n_bins / bins_per_octave)
    divided_by = 2 ** (num_repeats - 1)
    padding = (divided_by - original_hop_length % divided_by) % divided_by
    hop_length = original_hop_length + padding

    for path in paths:
        _waveform, _sample_rate = torchaudio.load(path)
        waveform.append(_waveform)

        if timesteps is None:
            timesteps = _waveform.size(-1)
        else:
            timesteps = min(timesteps, _waveform.size(-1))

        assert _sample_rate == sample_rate

    timesteps -= timesteps % (2**num_repeats)

    for idx in range(len(waveform)):
        waveform[idx], _ = torch.split(
            waveform[idx], [timesteps, waveform[idx].size(-1) - timesteps], dim=-1
        )

    waveform = torch.cat(waveform, dim=0)

    cqt = ConstantQTransform(
        sample_rate,
        hop_length,
        f_min=f_min,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        by_octave=False,
    )
    cqt_time = ConstantQTransform(
        sample_rate,
        hop_length,
        f_min=f_min,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        by_octave=True,
        domain="time",
        sparse=False,
    )
    cqt_freq = ConstantQTransform(
        sample_rate,
        hop_length,
        f_min=f_min,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        by_octave=True,
        domain="freq",
        sparse=False,
    )
    cqt_freq_sparse = ConstantQTransform(
        sample_rate,
        hop_length,
        f_min=f_min,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        by_octave=True,
        domain="freq",
        sparse=True,
    )

    spectrogram = cqt(waveform)
    spectrogram_time = cqt_time(waveform)
    spectrogram_freq = cqt_freq(waveform)
    spectrogram_freq_sparse = cqt_freq_sparse(waveform)

    assert spectrogram.size()[:2] == (batch_size, n_bins)

    if n_bins / bins_per_octave == 1:
        assert torch.allclose(spectrogram, spectrogram_time)
    else:
        # Computational error happens by resampling.
        assert torch.allclose(spectrogram, spectrogram_time, atol=1e-2)

    assert torch.allclose(spectrogram_time, spectrogram_freq, atol=1e-7)

    # Computational error happens by sparseness.
    assert torch.allclose(spectrogram_freq, spectrogram_freq_sparse, atol=1e-4)

    # Invalid cases
    if n_bins / bins_per_octave != 1:
        with pytest.raises(AssertionError) as e:
            _ = ConstantQTransform(
                sample_rate,
                original_hop_length,
                f_min=f_min,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                by_octave=True,
                domain="time",
                sparse=False,
            )

        assert str(e.value) == "Given hop length ({}) is not divisible by {}.".format(
            original_hop_length, divided_by
        )

    with pytest.raises(ValueError) as e:
        _ = ConstantQTransform(
            sample_rate,
            f_min,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            by_octave=False,
            domain="freq",
        )

    assert str(e.value) == "When domain='freq', only by_octave=True is supported."

    with pytest.raises(AssertionError) as e:
        _ = ConstantQTransform(
            sample_rate,
            f_min,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            by_octave=True,
            domain="time",
            sparse=True,
        )

    assert str(e.value) == "sparse=True is not supported when domain='time'."


def test_build_temporal_kernel() -> None:
    sample_rate = 16000
    f_min = 4000
    f_max = 8000
    n_bins = 13
    bins_per_octave = 12

    kernel_by_f_max = build_temporal_kernel(
        sample_rate,
        f_min,
        f_max=f_max,
        n_bins=n_bins,
        bins_per_octave=None,
    )

    kernel_by_bins_per_octave = build_temporal_kernel(
        sample_rate,
        f_min,
        f_max=None,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )

    kernel_by_default = build_temporal_kernel(
        sample_rate,
        f_min,
        f_max=None,
        n_bins=n_bins,
        bins_per_octave=None,
    )

    assert torch.equal(kernel_by_f_max, kernel_by_bins_per_octave)
    assert torch.equal(kernel_by_f_max, kernel_by_default)

    with pytest.raises(ValueError) as e:
        _ = build_temporal_kernel(
            sample_rate,
            f_min,
            f_max=f_max,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )

    assert str(e.value) == "Set either of f_max and bins_per_octave."


@pytest.mark.parametrize("_type", [int, list, torch.Tensor])
def test_compute_filter_length(_type: type) -> None:
    sample_rate = 16000
    bins_per_octave = 12
    scaling_factor = 1

    if _type is int:
        freqs = 8000
        expected_type = int
        expected_length = 34
    elif _type is list:
        freqs = [4000, 8000]
        expected_type = list
        expected_length = [68, 34]
    elif _type is torch.Tensor:
        freqs = torch.tensor([4000, 8000])
        expected_type = torch.Tensor
        expected_length = torch.tensor([68, 34])
    else:
        raise TypeError("Invalid type is given.")

    length = compute_filter_length(
        freqs,
        sample_rate,
        bins_per_octave=bins_per_octave,
        scaling_factor=scaling_factor,
    )

    assert type(length) is expected_type

    if _type is int or _type is list:
        assert length == expected_length
    elif _type is torch.Tensor:
        assert torch.equal(length, expected_length)
    else:
        raise TypeError("Invalid type is given.")
