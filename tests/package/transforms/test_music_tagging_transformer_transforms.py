import os
import tempfile

import torch
from dummy import allclose

from audyn.transforms.music_tagging_transformer import (
    MusicTaggingTransformerMelSpectrogram,
)
from audyn.utils.github import download_file_from_github_release


def test_music_tagging_transformer_melspectrogram() -> None:
    # regression test
    with tempfile.TemporaryDirectory() as temp_dir:
        url = "https://github.com/tky823/Audyn/releases/download/v0.0.2/test_official_music-tagging-transformer_transform.pth"  # noqa: E501
        path = os.path.join(temp_dir, "test_official_music-tagging-transformer_transform.pth")
        download_file_from_github_release(url, path)

        data = torch.load(path)

    waveform = data["input"]
    expected_melspectrogram = data["output"]

    melspectrogram_transform = MusicTaggingTransformerMelSpectrogram.build_from_pretrained()
    melspectrogram = melspectrogram_transform(waveform)

    allclose(melspectrogram, expected_melspectrogram)
