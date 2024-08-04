import os
import tempfile
from urllib.request import Request, urlopen

import torch
import torchaudio
from dummy import allclose

from audyn.transforms.music_tagging_transformer import MusicTaggingTransformerMelSpectrogram
from audyn.utils.github import download_file_from_github_release


def test_music_tagging_transformer_melspectrogram() -> None:
    n_fft = 1024

    audio_url = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"  # noqa: E501
    chunk_size = 8192

    # regression test
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.basename(audio_url)
        audio_path = os.path.join(temp_dir, filename)

        request = Request(audio_url)

        with urlopen(request) as response, open(audio_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)

                if not chunk:
                    break

                f.write(chunk)

        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)

        url = "https://github.com/tky823/Audyn/releases/download/v0.0.2/test_music_tagging_transformer_melspectrogram.pth"  # noqa: E501
        path = os.path.join(temp_dir, "test_music_tagging_transformer_melspectrogram.pth")
        download_file_from_github_release(url, path)

        data = torch.load(path)
        expected_output = data["output"]

    melspectrogram_transform = MusicTaggingTransformerMelSpectrogram(sample_rate, n_fft=n_fft)
    melspectrogram = melspectrogram_transform(waveform)

    allclose(melspectrogram, expected_output)
