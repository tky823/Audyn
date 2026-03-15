import os
import tempfile

import pytest
import torch
import torchvision
import torchvision.transforms.v2 as vT
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir
from packaging import version
from PIL import Image

from audyn.transforms.clip import OpenAICLIPImageTransform
from audyn.utils._github import download_file_from_github_release
from audyn.utils.data.download import download_file

IS_TORCHVISION_LT_0_16 = version.parse(torchvision.__version__) < version.parse("0.16")


def test_openai_clip_transform(image: Image) -> None:
    pretrained_model_name = "openai-clip-base-patch32"

    url = "https://github.com/tky823/Audyn/releases/download/v0.3.1/test_official_openai-clip.pth"  # noqa: E501

    filename = os.path.basename(url)
    path = os.path.join(audyn_test_cache_dir, filename)

    download_file_from_github_release(url, path)

    data = torch.load(path, weights_only=True)
    expected_output = data[pretrained_model_name]["transform"]

    if IS_TORCHVISION_LT_0_16:
        to_tensor = vT.PILToTensor()
    else:
        to_tensor = vT.ToImage()

    transform = OpenAICLIPImageTransform()
    output = transform(image)

    allclose(expected_output, output, atol=1e-6)

    image_torch = to_tensor(image)
    output_torch = transform(image_torch)

    loss = torch.abs(output - output_torch)

    assert loss.mean() < 1e-4


@pytest.fixture
def image() -> Image:
    with tempfile.TemporaryDirectory() as temp_dir:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        path = os.path.join(temp_dir, "000000039769.jpg")
        download_file(url, path)
        image = Image.open(path)
        image = image.copy()

    return image
