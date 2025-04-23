import os
import tempfile

import torch
import torch.nn as nn
from dummy import allclose

from audyn.models.clap import LAIONAudioEncoder2023, MicrosoftAudioEncoder2023
from audyn.utils._github import download_file_from_github_release


def test_official_laion_audio_encoder() -> None:
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.5/test_official_laion-clap-htsat-fused.pth"  # noqa: E501

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.basename(url)
        path = os.path.join(temp_dir, filename)
        download_file_from_github_release(url, path)

        data = torch.load(
            path,
            weights_only=True,
        )

    spectrogram = data["long"]["transform"]
    expected_output = data["long"]["output"]
    expected_embedding = data["long"]["embedding"]

    spectrogram = spectrogram.unsqueeze(dim=0)

    model = LAIONAudioEncoder2023.build_from_pretrained(
        "laion-clap-htsat-fused", aggregator=nn.Identity(), head=nn.Identity()
    )
    model.eval()

    with torch.no_grad():
        output = model(spectrogram)

    output = output.squeeze(dim=0)

    allclose(output, expected_output, atol=1e-5)

    model = LAIONAudioEncoder2023.build_from_pretrained("laion-clap-htsat-fused")
    model.eval()

    with torch.no_grad():
        embedding = model(spectrogram)

    embedding = embedding.squeeze(dim=0)

    allclose(embedding, expected_embedding, atol=1e-5)

    num_parameters = 0

    for p in model.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    assert num_parameters == 28205512


def test_official_microsoft_audio_encoder() -> None:
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.5/test_official_microsoft-clap-2023.pth"  # noqa: E501

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.basename(url)
        path = os.path.join(temp_dir, filename)
        download_file_from_github_release(url, path)

        data = torch.load(
            path,
            weights_only=True,
        )

    spectrogram = data["long"]["transform"]
    expected_output = data["long"]["output"]
    expected_embedding = data["long"]["embedding"]

    spectrogram = spectrogram.unsqueeze(dim=0)

    model = MicrosoftAudioEncoder2023.build_from_pretrained(
        "microsoft-clap-2023",
        aggregator=nn.Identity(),
        head=nn.Identity(),
    )
    model.eval()

    with torch.no_grad():
        output = model(spectrogram)

    output = output.squeeze(dim=0)

    allclose(output, expected_output, atol=1e-5)

    model = MicrosoftAudioEncoder2023.build_from_pretrained("microsoft-clap-2023")
    model.eval()

    with torch.no_grad():
        embedding = model(spectrogram)

    embedding = embedding.squeeze(dim=0)

    allclose(embedding, expected_embedding, atol=1e-6)

    num_parameters = 0

    for p in model.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    assert num_parameters == 29371544
