import os
import tempfile

import torch
from dummy import allclose

from audyn.transforms.clap import LAIONAudioEncoder2023MelSpectrogram
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
