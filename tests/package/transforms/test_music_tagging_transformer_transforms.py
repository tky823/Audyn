import os

import torch
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir

from audyn.transforms.music_tagging_transformer import (
    MusicTaggingTransformerMelSpectrogram,
)
from audyn.utils._github import download_file_from_github_release


def test_music_tagging_transformer_melspectrogram() -> None:
    # regression test
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.2/test_official_music-tagging-transformer_transform.pth"  # noqa: E501
    path = os.path.join(
        audyn_test_cache_dir, "test_official_music-tagging-transformer_transform.pth"
    )
    download_file_from_github_release(url, path)

    data = torch.load(path, weights_only=True)

    waveform = data["input"]
    expected_melspectrogram = data["output"]

    melspectrogram_transform = MusicTaggingTransformerMelSpectrogram.build_from_pretrained()
    melspectrogram = melspectrogram_transform(waveform)

    allclose(melspectrogram, expected_melspectrogram, atol=1e-4)
