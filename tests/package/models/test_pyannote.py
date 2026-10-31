import os

import torch
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir

from audyn.models.pyannote import PyannoteCompatibleSpeakerSegmenter
from audyn.modules.diarization import PowerSet
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
    expected_activity = data["activity"]

    num_speakers = 3

    model = PyannoteCompatibleSpeakerSegmenter.build_from_pretrained("pyannote-3.0")
    to_multilabel = PowerSet(num_classes=num_speakers, max_subset=2)

    input = input.view(1, 1, -1)

    model.eval()

    with torch.inference_mode():
        output = model(input)

    activity = to_multilabel(output)

    allclose(expected_output, output.squeeze(dim=0), atol=1e-3)
    assert torch.equal(activity.squeeze(dim=0), expected_activity.to(activity.dtype))
