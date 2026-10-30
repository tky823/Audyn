import os

import torch
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir

from audyn.modules.filterbanks import QuadratureSincConv1d
from audyn.utils._github import download_file_from_github_release


def test_quadrature_sinc_conv1d() -> None:
    url = "https://github.com/tky823/Audyn/releases/download/v0.3.1/test_filterbanks.pth"

    filename = os.path.basename(url)
    path = os.path.join(audyn_test_cache_dir, filename)
    download_file_from_github_release(url, path)

    n_bins = 8
    kernel_size = 31

    module = QuadratureSincConv1d(n_bins, kernel_size=kernel_size)

    data = torch.load(path, weights_only=True)
    input = data["quadrature_sinc_conv1d"]["input"]
    expected_output = data["quadrature_sinc_conv1d"]["output"]

    output = module(input.unsqueeze(dim=0))

    allclose(output, expected_output, atol=1e-5)
