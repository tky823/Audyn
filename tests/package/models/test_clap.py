import os
import tempfile

import torch
from dummy import allclose

from audyn.models.clap import LAIONAudioEncoder2023
from audyn.utils._github import download_file_from_github_release


def test_official_laion_audio_encoder() -> None:
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.5/test_official_laion-clap-htsat-fused.pth"  # noqa: E501
    model = LAIONAudioEncoder2023.build_from_pretrained("laion-clap-htsat-fused")

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.basename(url)
        path = os.path.join(temp_dir, filename)
        download_file_from_github_release(url, path)

        data = torch.load(path, weights_only=True)

    spectrogram = data["long"]["transform"]
    expected_output = data["long"]["output"]

    spectrogram = spectrogram.unsqueeze(dim=0)

    model.eval()

    with torch.no_grad():
        output = model(spectrogram)

    output = output.squeeze(dim=0)

    allclose(output, expected_output, atol=1e-5)

    num_parameters = 0

    for p in model.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    assert num_parameters == 27549128
