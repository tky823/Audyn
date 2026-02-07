import os
import tempfile
from typing import Any, Dict

import pytest
import torch
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir
from PIL import Image

from audyn.utils._github import download_file_from_github_release
from audyn.utils.data.clip import OpenAIImageEncoderComposer
from audyn.utils.data.download import download_file


def test_openai_image_encoder_composer(coco_samples) -> None:
    url = "https://github.com/tky823/Audyn/releases/download/v0.1.0/test_official_openai-clip.pth"  # noqa: E501

    filename = os.path.basename(url)
    path = os.path.join(audyn_test_cache_dir, filename)
    download_file_from_github_release(url, path)

    composer = OpenAIImageEncoderComposer(input_key="image", output_key="image")

    keys = sorted(coco_samples.keys())
    list_batch = []

    for key in keys:
        list_batch.append(coco_samples[key])

    list_batch = composer(list_batch)

    for sample in list_batch:
        expected_sample = torch.load(path, weights_only=True)
        allclose(sample["image"], expected_sample["input"])

        break


@pytest.fixture
def coco_samples() -> Dict[str, Dict[str, Any]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        path = os.path.join(temp_dir, "000000039769.jpg")
        download_file(url, path)
        image0 = Image.open(path)
        image0 = image0.copy()

    samples = {
        "example0": {
            "image": image0,
        },
    }

    return samples
