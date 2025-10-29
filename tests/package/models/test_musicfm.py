import os

import torch
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir

from audyn.models.musicfm import MusicFM
from audyn.utils._github import download_file_from_github_release


def test_musicfm() -> None:
    # regression test
    url = "https://github.com/tky823/Audyn/releases/download/v0.2.0/test_official_musicfm.pth"  # noqa: E501
    path = os.path.join(audyn_test_cache_dir, "test_official_musicfm.pth")
    download_file_from_github_release(url, path)

    data = torch.load(path, weights_only=True)

    spectrogram = data["spectrogram"]
    expected_logits = data["msd"]["logits"]

    model = MusicFM.build_from_pretrained("musicfm_msd")

    model.eval()

    with torch.no_grad():
        logits = model(spectrogram)

    allclose(logits, expected_logits, atol=1e-5)
