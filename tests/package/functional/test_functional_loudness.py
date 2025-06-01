import os

import torch
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir

from audyn.functional.loudness import compute_loudness
from audyn.utils._github import download_file_from_github_release


def test_compute_loudness() -> None:
    url = "https://github.com/tky823/Audyn/releases/download/v0.1.0/test_loudness.pth"
    path = os.path.join(audyn_test_cache_dir, "test_loudness.pth")
    download_file_from_github_release(url, path)

    data = torch.load(path, weights_only=True)
    waveform = data["waveform"]
    sample_rate = data["sample_rate"]
    expected_loudness = data["loudness"]

    loudness = compute_loudness(waveform, sample_rate=sample_rate)

    allclose(loudness, expected_loudness)
