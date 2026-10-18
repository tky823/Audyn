import os

import pytest
import torch
import torch.nn as nn
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir

from audyn.models import T5Encoder, T5EncoderLayer, T5TransformerEncoder
from audyn.utils._github import download_file_from_github_release


@pytest.mark.parametrize("batch_first", [True, False])
def test_t5_encoder(batch_first: bool) -> None:
    embedding_dim = 8
    batch_size = 2
    length = 5

    embedding = nn.Embedding(32, embedding_dim)
    layer = T5EncoderLayer(
        d_model=embedding_dim,
        nhead=2,
        dim_feedforward=16,
        dropout=0,
        batch_first=batch_first,
    )
    backbone = T5TransformerEncoder(layer, num_layers=2)
    model = T5Encoder(embedding, backbone, batch_first=batch_first)
    model.eval()

    if batch_first:
        input = torch.randint(0, 32, (batch_size, length))
        expected_shape = (batch_size, length, embedding_dim)
    else:
        input = torch.randint(0, 32, (length, batch_size))
        expected_shape = (length, batch_size, embedding_dim)

    padding_mask = torch.tensor(
        [[False, False, False, False, True], [False, False, True, True, True]]
    )

    with torch.no_grad():
        output = model(input, padding_mask=padding_mask)

    assert output.size() == expected_shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("pretrained_model_name", ["t5-base", "t5-large"])
def test_official_t5_encoder(pretrained_model_name: str) -> None:
    url = "https://github.com/tky823/Audyn/releases/download/v0.3.2/test_t5.pth"

    filename = os.path.basename(url)
    path = os.path.join(audyn_test_cache_dir, filename)
    download_file_from_github_release(url, path)

    data = torch.load(
        path,
        weights_only=True,
    )

    input = data[pretrained_model_name]["input"]
    padding_mask = data[pretrained_model_name]["padding_mask"]
    expected_output = data[pretrained_model_name]["output"]

    model = T5Encoder.build_from_pretrained(pretrained_model_name)
    model.eval()

    with torch.no_grad():
        output = model(input, padding_mask=padding_mask)

    allclose(output, expected_output, atol=1e-4)
