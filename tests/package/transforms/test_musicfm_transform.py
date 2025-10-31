import os

import torch
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir

from audyn.transforms.musicfm import MusicFMMelSpectrogram
from audyn.utils._github import download_file_from_github_release


def test_musicfm_melspectrogram_transform() -> None:
    # regression test
    url = "https://github.com/tky823/Audyn/releases/download/v0.2.0/test_official_musicfm.pth"  # noqa: E501
    path = os.path.join(audyn_test_cache_dir, "test_official_musicfm.pth")
    download_file_from_github_release(url, path)

    data = torch.load(path, weights_only=True)

    waveform = data["waveform"]
    expected_melspectrogram = data["spectrogram"]

    transform = MusicFMMelSpectrogram.build_from_pretrained()
    melspectrogram = transform(waveform)

    allclose(melspectrogram, expected_melspectrogram, atol=1e-4)
