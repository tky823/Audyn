import os
import tempfile

import pytest
import torch
from dummy import allclose

from audyn.transforms.clap import (
    LAIONAudioEncoder2023MelSpectrogram,
    LAIONAudioEncoder2023MelSpectrogramFusion,
    LAIONAudioEncoder2023WaveformPad,
)
from audyn.utils._github import download_file_from_github_release


def test_laion_clap_transform() -> None:
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.5/test_official_laion-clap-htsat-fused.pth"  # noqa: E501

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.basename(url)
        path = os.path.join(temp_dir, filename)

        download_file_from_github_release(url, path)

        data = torch.load(path, weights_only=True)

    padding_transform = LAIONAudioEncoder2023WaveformPad.build_from_pretrained(
        "laion-clap-htsat-fused"
    )
    melspectrogram_transform = LAIONAudioEncoder2023MelSpectrogram.build_from_pretrained(
        "laion-clap-htsat-fused"
    )
    fusion_transform = LAIONAudioEncoder2023MelSpectrogramFusion.build_from_pretrained(
        "laion-clap-htsat-fused"
    )
    melspectrogram_transform.eval()
    fusion_transform.eval()

    # waveform longer than chunk_size
    waveform = data["long"]["input"]
    expected_output = data["long"]["transform"]
    waveform = padding_transform(waveform)
    melspectrogram = melspectrogram_transform(waveform)
    output = fusion_transform(melspectrogram)

    allclose(output, expected_output, atol=1e-5)

    # waveform shorter than chunk_size
    waveform = data["short"]["input"]
    expected_output = data["short"]["transform"]
    waveform = padding_transform(waveform)
    melspectrogram = melspectrogram_transform(waveform)
    output = fusion_transform(melspectrogram)

    allclose(output, expected_output, atol=1e-5)


@pytest.mark.parametrize("fb_dtype", [None, torch.float64])
def test_laion_clap_melspectrogram(fb_dtype: torch.dtype) -> None:  # noqa: E501
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.5/test_official_laion-clap-htsat-fused.pth"  # noqa: E501

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.basename(url)
        path = os.path.join(temp_dir, filename)

        download_file_from_github_release(url, path)

        data = torch.load(path, weights_only=True)

    waveform = data["middle"]["input"]
    sample_rate = data["sample_rate"]

    # htk
    melspectrogram_transform = LAIONAudioEncoder2023MelSpectrogram(
        sample_rate,
        norm=None,
        mel_scale="htk",
        fb_dtype=fb_dtype,
    )
    output = melspectrogram_transform(waveform)
    expected_output = data["middle"]["transform"]["htk"]

    if fb_dtype is torch.float64:
        # test only high-precision
        allclose(output, expected_output, atol=1e-5)

    # slaney
    melspectrogram_transform = LAIONAudioEncoder2023MelSpectrogram(
        sample_rate,
        norm="slaney",
        mel_scale="slaney",
        fb_dtype=fb_dtype,
    )
    output = melspectrogram_transform(waveform)
    expected_output = data["middle"]["transform"]["slaney"]

    if fb_dtype is torch.float64:
        # test only high-precision
        allclose(output, expected_output, atol=1e-7)


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

    with pytest.warns(
        UserWarning,
        match=f"Number of frames {short_melspectrogram.size(-1)} is shorter "
        f"than required chunk_size {chunk_size}.",
    ):
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
