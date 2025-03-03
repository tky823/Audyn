import os
import tempfile

import pytest
import torch
from dummy import allclose

from audyn.transforms.clap import (
    LAIONAudioEncoder2023MelSpectrogram,
    LAIONAudioEncoder2023MelSpectrogramFusion,
)
from audyn.utils._github import download_file_from_github_release


def test_laion_clap_melspectrogram() -> None:
    # NOTE: High band may contain larger errors due to numerical precision.
    high_band_idx = 60

    sample_rate = 48000
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.4/test_laion_clap_melspectrogram.pth"  # noqa: E501

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.basename(url)
        path = os.path.join(temp_dir, filename)

        download_file_from_github_release(url, path)

        data = torch.load(path)

    waveform = data["input"]

    # htk
    melspectrogram_transform = LAIONAudioEncoder2023MelSpectrogram(
        sample_rate, norm=None, mel_scale="htk"
    )
    melspectrogram = melspectrogram_transform(waveform)
    expected_output = data["htk"]["output"]
    error = torch.abs(melspectrogram - expected_output)
    mean_error = error.mean()
    mean_error = mean_error.item()

    allclose(melspectrogram[:high_band_idx], expected_output[:high_band_idx], atol=1e-3)
    assert mean_error < 1e-4

    # slaney
    melspectrogram_transform = LAIONAudioEncoder2023MelSpectrogram(
        sample_rate, norm="slaney", mel_scale="slaney"
    )
    melspectrogram = melspectrogram_transform(waveform)
    expected_output = data["slaney"]["output"]
    error = torch.abs(melspectrogram - expected_output)
    mean_error = error.mean()
    mean_error = mean_error.item()

    allclose(melspectrogram[:high_band_idx], expected_output[:high_band_idx], atol=1e-3)
    assert mean_error < 1e-4


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("prepend_resampled_chunk", [True, False])
@pytest.mark.parametrize("pad_mode", ["replicate+constant", "replicate"])
@pytest.mark.parametrize("sample_wise", [True, False])
def test_laion_clap_melspectrogram_fusion(
    center: bool, prepend_resampled_chunk: bool, pad_mode: str, sample_wise: bool
) -> None:
    torch.manual_seed(0)

    batch_shape = (4, 2)
    sample_rate = 32000
    long_duration = 6
    short_duration = 3
    chunk_duration = 4

    long_length = int(sample_rate * long_duration)
    short_length = int(sample_rate * short_duration)
    long_waveform = torch.randn((*batch_shape, long_length))
    short_waveform = torch.randn((*batch_shape, short_length))

    melspectrogram_transform = LAIONAudioEncoder2023MelSpectrogram(
        sample_rate, center=center, norm="slaney", mel_scale="slaney"
    )
    win_length = melspectrogram_transform.win_length
    hop_length = melspectrogram_transform.hop_length
    chunk_length = int(sample_rate * chunk_duration)
    waveform = torch.randn((*batch_shape, chunk_length))

    if center:
        chunk_size = (chunk_length - win_length + 2 * (win_length // 2)) // hop_length + 1
    else:
        chunk_size = (chunk_length - win_length) // hop_length + 1

    fusion_transform = LAIONAudioEncoder2023MelSpectrogramFusion(
        chunk_size=chunk_size,
        prepend_resampled_chunk=prepend_resampled_chunk,
        pad_mode=pad_mode,
        sample_wise=sample_wise,
    )

    melspectrogram = melspectrogram_transform(waveform)
    long_melspectrogram = melspectrogram_transform(long_waveform)
    short_melspectrogram = melspectrogram_transform(short_waveform)

    fused_melspectrogram = fusion_transform(melspectrogram)
    long_melspectrogram = fusion_transform(long_melspectrogram)
    short_melspectrogram = fusion_transform(short_melspectrogram)

    assert long_melspectrogram.size()[-2:] == melspectrogram.size()[-2:]
    assert (
        long_melspectrogram.size()[: len(batch_shape)] == melspectrogram.size()[: len(batch_shape)]
    )
    assert short_melspectrogram.size()[-2:] == melspectrogram.size()[-2:]
    assert (
        short_melspectrogram.size()[: len(batch_shape)]
        == melspectrogram.size()[: len(batch_shape)]
    )

    if prepend_resampled_chunk:
        assert long_melspectrogram.size(-3) == fusion_transform.num_chunks + 1
        assert short_melspectrogram.size(-3) == fusion_transform.num_chunks + 1
    else:
        assert long_melspectrogram.size(-3) == fusion_transform.num_chunks
        assert short_melspectrogram.size(-3) == fusion_transform.num_chunks

    fused_melspectrograms = torch.unbind(fused_melspectrogram, dim=-3)

    for fused_melspectrogram in fused_melspectrograms:
        allclose(fused_melspectrogram, melspectrogram)
