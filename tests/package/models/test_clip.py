import os

import torch
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir

from audyn.models.clip import OpenAICLIPImageEncoder
from audyn.utils._github import download_file_from_github_release


def test_official_openai_clip_image_encoder() -> None:
    pretrained_model_name = "openai-clip-base-patch32"

    url = "https://github.com/tky823/Audyn/releases/download/v0.3.1/test_official_openai-clip.pth"  # noqa: E501

    filename = os.path.basename(url)
    path = os.path.join(audyn_test_cache_dir, filename)
    download_file_from_github_release(url, path)

    data = torch.load(path, weights_only=True)

    input = data[pretrained_model_name]["transform"]
    expected_output = data[pretrained_model_name]["output"]

    input = input.unsqueeze(dim=0)

    model = OpenAICLIPImageEncoder.build_from_pretrained(pretrained_model_name)
    model.eval()

    with torch.no_grad():
        output = model(input)

    output = output.squeeze(dim=0)

    allclose(output, expected_output, atol=1e-5)

    num_parameters = 0

    for p in model.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    assert num_parameters == 87848448
