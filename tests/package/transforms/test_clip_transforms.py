import os
import tempfile

import torch
import torchvision.transforms.v2 as vT
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir
from PIL import Image

from audyn.transforms.clip import OpenAICLIPImageTransform
from audyn.utils._github import download_file_from_github_release


def test_openai_clip_transform() -> None:
    pretrained_model_name = "openai-clip-base-patch32"

    url = "https://github.com/tky823/Audyn/releases/download/v0.3.1/test_official_openai-clip.pth"  # noqa: E501

    filename = os.path.basename(url)
    path = os.path.join(audyn_test_cache_dir, filename)

    download_file_from_github_release(url, path)

    data = torch.load(path, weights_only=True)

    image_bytes = data["input"]
    expected_output = data[pretrained_model_name]["transform"]

    to_tensor = vT.Compose([vT.ToImage()])
    transform = OpenAICLIPImageTransform()

    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, "input.jpg")

        with open(image_path, mode="wb") as f:
            f.write(image_bytes)

        image = Image.open(image_path)

    output = transform(image)

    allclose(expected_output, output, atol=1e-6)

    image_torch = to_tensor(image)
    output_torch = transform(image_torch)

    loss = torch.abs(output - output_torch)

    assert loss.mean() < 1e-4
