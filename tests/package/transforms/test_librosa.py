import os
import tempfile

import torch
import torchaudio
from dummy import allclose

from audyn.transforms.librosa import LibrosaMelSpectrogram
from audyn.utils.github import download_file_from_github_release


def test_librosa_melspectrogram() -> None:

    # regression test
    path = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"  # noqa: E501
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)

    with tempfile.TemporaryDirectory() as temp_dir:
        url = "https://github.com/tky823/Audyn/releases/download/v0.0.1.dev4/test_librosa_melspectrogram.pth"  # noqa: E501
        path = os.path.join(temp_dir, "test_librosa_melspectrogram.pth")
        download_file_from_github_release(url, path)

        data = torch.load(path)
        expected_output = data["output"]

    melspectrogram_transform = LibrosaMelSpectrogram(sample_rate)
    melspectrogram = melspectrogram_transform(waveform)

    allclose(melspectrogram, expected_output, atol=1e-4)
