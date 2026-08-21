import os

import torch
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir

from audyn.transforms.muq import MuQMelSpectrogram
from audyn.utils._github import download_file_from_github_release


def test_muq_melspectrogram_transform() -> None:
    # regression test
    url = "https://github.com/tky823/Audyn/releases/download/v0.3.0/test_official_muq.pth"  # noqa: E501
    path = os.path.join(audyn_test_cache_dir, "test_official_muq.pth")
    download_file_from_github_release(url, path)

    data = torch.load(path, weights_only=True)

    waveform = data["waveform"]

    # MSD
    dataset = "msd"
    expected_melspectrogram = data[dataset]["spectrogram"]

    transform = MuQMelSpectrogram.build_from_pretrained(dataset=dataset)
    melspectrogram = transform(waveform)

    allclose(melspectrogram, expected_melspectrogram, atol=1e-6)
