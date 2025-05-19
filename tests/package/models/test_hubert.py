import os
import tempfile

import torch
from audyn_test import allclose

from audyn.models.hubert import HuBERT
from audyn.utils._github import download_file_from_github_release


def test_hubert() -> None:
    model = HuBERT.build_from_pretrained("hubert-large-librispeech960-finetuning")

    with tempfile.TemporaryDirectory() as temp_dir:
        url = "https://github.com/tky823/Audyn/releases/download/v0.0.6/test_official_hubert.pth"
        path = os.path.join(temp_dir, "test_official_hubert.pth")
        download_file_from_github_release(url, path)

        data = torch.load(
            path,
            weights_only=True,
        )
        waveform = data["input"]
        expected_output = data["output"]
        expected_embedding = data["embedding"]

    model.eval()

    waveform = waveform.unsqueeze(dim=0)

    with torch.no_grad():
        embedding = model.embedding(waveform)

    embedding = embedding.squeeze(dim=0)

    allclose(embedding, expected_embedding, atol=1e-4)

    embedding = expected_embedding.unsqueeze(dim=0)

    with torch.no_grad():
        output = model.backbone(embedding)

    output = output.squeeze(dim=0)

    allclose(output, expected_output, atol=1e-3)

    with torch.no_grad():
        output = model(waveform)

    output = output.squeeze(dim=0)

    allclose(output, expected_output, atol=1e-3)

    model.remove_weight_norm_()
