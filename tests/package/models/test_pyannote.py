import os

import torch
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir

from audyn.models.pyannote import PyannoteCompatibleSpeakerSegmenter
from audyn.utils._github import download_file_from_github_release


def test_official_speaker_segmenter() -> None:
    url = "https://github.com/tky823/Audyn/releases/download/v0.3.1/test_pyannote-3.0.pth"

    filename = os.path.basename(url)
    path = os.path.join(audyn_test_cache_dir, filename)
    download_file_from_github_release(url, path)

    data = torch.load(
        path,
        weights_only=True,
    )

    input = data["input"]
    expected_output = data["output"]

    model = PyannoteCompatibleSpeakerSegmenter.build_from_pretrained("pyannote-3.0")

    input = data["input"]
    expected_output = data["output"]

    input = input.view(1, 1, -1)

    model.eval()

    with torch.inference_mode():
        output = model(input)

    allclose(expected_output, output.squeeze(dim=0), atol=1e-3)
