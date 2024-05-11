import os
import tempfile

import pytest
import torch
import torch.nn as nn

from audyn.models.ast import (
    AudioSpectrogramTransformer,
    HeadTokensAggregator,
    MLPHead,
    PositionalPatchEmbedding,
)
from audyn.utils.github import download_file_from_github_release


def test_official_ast() -> None:
    torch.manual_seed(0)

    d_model, out_channels = 768, 527
    n_bins, n_frames = 128, 1024
    kernel_size = (16, 16)
    stride = (10, 10)

    insert_cls_token = True
    insert_dist_token = True

    nhead = 12
    dim_feedforward = 3072
    num_layers = 12

    expected_num_parameters = 86592527

    patch_embedding = PositionalPatchEmbedding(
        d_model,
        kernel_size=kernel_size,
        stride=stride,
        insert_cls_token=insert_cls_token,
        insert_dist_token=insert_dist_token,
        n_bins=n_bins,
        n_frames=n_frames,
    )
    encoder_layer = nn.TransformerEncoderLayer(
        d_model,
        nhead,
        dim_feedforward=dim_feedforward,
        activation=nn.GELU(),
        batch_first=True,
    )
    norm = nn.LayerNorm(d_model)
    transformer = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_layers,
        norm=norm,
    )
    aggregator = HeadTokensAggregator(
        insert_cls_token=insert_cls_token, insert_dist_token=insert_dist_token
    )
    head = MLPHead(d_model, out_channels)
    model = AudioSpectrogramTransformer(
        patch_embedding,
        transformer,
        aggregator=aggregator,
        head=head,
    )

    num_parameters = 0

    for p in model.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    assert num_parameters == expected_num_parameters

    # build_from_pretrained
    model = AudioSpectrogramTransformer.build_from_pretrained(
        "ast-base-stride10",
        stride=stride,
        n_bins=n_bins,
        n_frames=n_frames,
        aggregator=aggregator,
        head=head,
    )

    num_parameters = 0

    for p in model.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    assert num_parameters == expected_num_parameters

    # regression test
    n_bins, n_frames = 256, 100
    model = AudioSpectrogramTransformer.build_from_pretrained(
        "ast-base-stride10",
        stride=stride,
        n_bins=n_bins,
        n_frames=n_frames,
        aggregator=aggregator,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        url = "https://github.com/tky823/Audyn/releases/download/v0.0.1.dev3/test_official_ast.pth"
        path = os.path.join(temp_dir, "test_official_ast.pth")
        download_file_from_github_release(url, path)

        data = torch.load(path)
        input = data["input"]
        expected_output = data["output"]

    model.eval()

    with torch.no_grad():
        output = model(input)

    assert torch.allclose(output, expected_output)


@pytest.mark.parametrize("take_length", [True, False])
def test_ast(take_length: bool) -> None:
    torch.manual_seed(0)

    d_model, out_channels = 8, 10
    n_bins, n_frames = 8, 30
    kernel_size = (4, 4)
    insert_cls_token, insert_dist_token = True, True
    batch_size = 4

    nhead = 2
    dim_feedforward = 5
    num_layers = 2

    patch_embedding = PositionalPatchEmbedding(
        d_model,
        kernel_size=kernel_size,
        insert_cls_token=insert_cls_token,
        insert_dist_token=insert_dist_token,
        n_bins=n_bins,
        n_frames=n_frames,
    )
    norm = nn.LayerNorm(d_model)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True
    )
    transformer = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_layers,
        norm=norm,
    )
    aggregator = HeadTokensAggregator(
        insert_cls_token=insert_cls_token,
        insert_dist_token=insert_dist_token,
    )
    head = MLPHead(d_model, out_channels)

    model = AudioSpectrogramTransformer(
        patch_embedding, transformer, aggregator=aggregator, head=head
    )

    if take_length:
        length = torch.randint(n_frames // 2, n_frames, (batch_size,), dtype=torch.long)
        input = torch.randn((batch_size, n_bins, n_frames))
        output = model(input, length=length)
    else:
        input = torch.randn((batch_size, n_bins, n_frames))
        output = model(input)

    assert output.size() == (batch_size, out_channels)


def test_ast_positional_patch_embedding() -> None:
    torch.manual_seed(0)

    d_model = 8
    n_bins, n_frames = 8, 30
    kernel_size = (n_bins, 2)
    batch_size = 4

    model = PositionalPatchEmbedding(
        d_model,
        kernel_size=kernel_size,
        n_bins=n_bins,
        n_frames=n_frames,
    )

    input = torch.randn((batch_size, n_bins, n_frames))
    output = model(input)

    Kh, Kw = kernel_size
    Sh, Sw = kernel_size
    height = (n_bins - Kh) // Sh + 1
    width = (n_frames - Kw) // Sw + 1

    assert output.size() == (batch_size, height * width, d_model)
