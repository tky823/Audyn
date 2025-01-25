import os
import tempfile

import torch
from dummy import allclose

from audyn.models.clap import LAIONAudioEncoder2023
from audyn.utils._github import download_file_from_github_release


def test_official_laion_audio_encoder() -> None:
    model = LAIONAudioEncoder2023.build_from_pretrained("laion-clap-htsat-fused")
    model.eval()

    with tempfile.TemporaryDirectory() as temp_dir:
        url = "https://github.com/tky823/Audyn/releases/download/v0.0.4/test_official_laion-clap-htsat-fused.pth"  # noqa: E501
        path = os.path.join(temp_dir, "test_official_laion-clap-htsat-fused.pth")
        download_file_from_github_release(url, path)

        data = torch.load(path)

    spectrogram = data["input"]
    expected_output = data["output"]

    with torch.no_grad():
        output = model(spectrogram)

    allclose(output, expected_output, atol=1e-4)
    assert torch.abs(output - expected_output) < 1e-6
