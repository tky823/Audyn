import os
import tempfile

import torch
from dummy import allclose

from audyn.models.clap import LAIONAudioEncoder2023
from audyn.utils._github import download_file_from_github_release


def test_official_laion_audio_encoder() -> None:
    model = LAIONAudioEncoder2023.build_from_pretrained("laion-clap-htsat-fused")

    with tempfile.TemporaryDirectory() as temp_dir:
        url = "https://github.com/tky823/Audyn/releases/download/v0.0.4/test_official_laion-clap-htsat-fused.pth"  # noqa: E501
        path = os.path.join(temp_dir, "test_official_laion-clap-htsat-fused.pth")
        download_file_from_github_release(url, path)

        data = torch.load(path)

    spectrogram = data["input"]
    expected_output = data["output"]

    model.eval()

    with torch.no_grad():
        output = model(spectrogram)

    error = torch.abs(output - expected_output)
    mean_error = error.mean()
    mean_error = mean_error.item()

    allclose(output, expected_output, atol=1e-3)
    assert mean_error < 1e-5

    num_parameters = 0

    for p in model.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    assert num_parameters == 27534488
