import os
import tempfile

import pytest
import torch
import torch.nn as nn
from dummy import allclose

from audyn.models.ast import (
    AST,
    AudioSpectrogramTransformer,
    AverageAggregator,
    HeadTokensAggregator,
    MLPHead,
)
from audyn.models.lextransformer import LEXTransformerEncoderLayer
from audyn.models.roformer import RoFormerEncoderLayer
from audyn.modules.vit import PatchEmbedding, PositionalPatchEmbedding
from audyn.utils._github import download_file_from_github_release


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
    assert model.__class__.__name__ == "AudioSpectrogramTransformer"

    # build_from_pretrained
    model = AST.build_from_pretrained(
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
    assert model.__class__.__name__ == "AST"

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

        data = torch.load(
            path,
            weights_only=True,
        )
        input = data["input"]
        expected_output = data["output"]

    model.eval()

    with torch.no_grad():
        output = model(input)

    allclose(output, expected_output)


def test_ast() -> None:
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

    input = torch.randn((batch_size, n_bins, n_frames))
    output = model(input)

    assert output.size() == (batch_size, out_channels)


@pytest.mark.parametrize("backbone", ["roformer", "lextransformer"])
@pytest.mark.parametrize("aggregation", ["head_tokens", "average"])
def test_ast_length(backbone: str, aggregation: str) -> None:
    torch.manual_seed(0)

    d_model, out_channels = 8, 10
    n_bins, n_frames = 8, 20
    kernel_size = (n_bins, 4)
    insert_cls_token, insert_dist_token = True, True
    batch_size = 4

    nhead = 2
    dim_feedforward = 5
    num_layers = 6

    patch_embedding = PatchEmbedding(
        d_model,
        kernel_size=kernel_size,
        insert_cls_token=insert_cls_token,
        insert_dist_token=insert_dist_token,
        n_bins=n_bins,
        n_frames=n_frames,
    )
    norm = nn.LayerNorm(d_model)

    if backbone == "roformer":
        encoder_layer = RoFormerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
    elif backbone == "lextransformer":
        encoder_layer = LEXTransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
    else:
        raise ValueError(f"{backbone} is not supported as backbone.")

    transformer = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_layers,
        norm=norm,
    )

    if aggregation == "head_tokens":
        aggregator = HeadTokensAggregator(
            insert_cls_token=insert_cls_token,
            insert_dist_token=insert_dist_token,
        )
    elif aggregation == "average":
        aggregator = AverageAggregator(
            insert_cls_token=insert_cls_token,
            insert_dist_token=insert_dist_token,
        )
    else:
        raise ValueError(f"{aggregation} is not supported as aggregation.")

    head = MLPHead(d_model, out_channels)
    model = AudioSpectrogramTransformer(
        patch_embedding, transformer, aggregator=aggregator, head=head
    )
    model.eval()

    # ensure invariance of padding
    with torch.no_grad():
        length = torch.randint(n_frames // 2, n_frames, (batch_size,), dtype=torch.long)
        longer_input = torch.randn((batch_size, n_bins, 2 * n_frames))

        padding_mask = torch.arange(2 * n_frames) >= length.unsqueeze(dim=-1)
        longer_input = longer_input.masked_fill(padding_mask.unsqueeze(dim=-2), 0)
        longer_output = model(longer_input, length=length)

        input, _ = torch.split(longer_input, [n_frames, n_frames], dim=-1)
        output = model(input, length=length)

    allclose(output, longer_output, atol=1e-6)


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
