import os
import tempfile

import pytest
import torch
import torch.nn as nn
from dummy import allclose

from audyn.models.ast import AverageAggregator
from audyn.models.ssast import (
    MLP,
    FastMasker,
    Masker,
    MLPHead,
    MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel,
    PositionalPatchEmbedding,
    SelfSupervisedAudioSpectrogramTransformer,
)
from audyn.utils.github import download_file_from_github_release


@pytest.mark.parametrize(
    "model_name",
    [
        "multitask-ssast-patch-base-400",
        "multitask-ssast-frame-base-400",
    ],
)
@pytest.mark.parametrize("sample_wise", [True, False])
def test_official_ssast_multi_task_mpm(model_name: str, sample_wise: bool) -> None:
    torch.manual_seed(0)

    d_model = 768
    n_bins, n_frames = 128, 1024

    if model_name == "multitask-ssast-patch-base-400":
        kernel_size = (16, 16)
    elif model_name == "multitask-ssast-frame-base-400":
        kernel_size = (n_bins, 2)
    else:
        raise ValueError("Invalid model name is given.")

    insert_cls_token = True
    insert_dist_token = True

    num_masks = 400
    min_cluster, max_cluster = 3, 6

    nhead = 12
    dim_feedforward = 3072
    num_layers = 12

    expected_num_parameters = 87223808

    patch_embedding = PositionalPatchEmbedding(
        d_model,
        kernel_size=kernel_size,
        insert_cls_token=insert_cls_token,
        insert_dist_token=insert_dist_token,
        n_bins=n_bins,
        n_frames=n_frames,
    )
    masker = Masker(
        d_model,
        num_masks=num_masks,
        min_cluster=min_cluster,
        max_cluster=max_cluster,
        sample_wise=sample_wise,
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
    reconstructor = MLP(d_model, kernel_size[0] * kernel_size[1])
    classifier = MLP(d_model, kernel_size[0] * kernel_size[1])
    model = MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel(
        patch_embedding,
        masker,
        transformer,
        reconstructor=reconstructor,
        classifier=classifier,
    )

    num_parameters = 0

    for p in model.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    assert num_parameters == expected_num_parameters

    model = (
        MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel.build_from_pretrained(
            model_name
        )
    )

    num_parameters = 0

    for p in model.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    assert num_parameters == expected_num_parameters


@pytest.mark.parametrize(
    "model_name",
    [
        "multitask-ssast-patch-base-400",
        "multitask-ssast-frame-base-400",
    ],
)
def test_official_ssast(model_name: str) -> None:
    torch.manual_seed(0)

    d_model, out_channels = 768, 35
    n_bins, n_frames = 128, 100

    if model_name == "multitask-ssast-patch-base-400":
        kernel_size = (16, 16)
        stride = (10, 10)
        expected_num_parameters = 85366307
    elif model_name == "multitask-ssast-frame-base-400":
        kernel_size = (n_bins, 2)
        stride = (n_bins, 1)
        expected_num_parameters = 85359395
    else:
        raise ValueError("Invalid model name is given.")

    insert_cls_token = True
    insert_dist_token = True

    nhead = 12
    dim_feedforward = 3072
    num_layers = 12

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
    aggregator = AverageAggregator()
    head = MLPHead(d_model, out_channels)
    model = SelfSupervisedAudioSpectrogramTransformer(
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
    model = SelfSupervisedAudioSpectrogramTransformer.build_from_pretrained(
        model_name, stride=stride, n_bins=n_bins, n_frames=n_frames, head=head
    )

    num_parameters = 0

    for p in model.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    assert num_parameters == expected_num_parameters

    # regression test
    n_bins, n_frames = 256, 100

    if model_name == "multitask-ssast-patch-base-400":
        filename = "test_official_ssast_patch.pth"
        stride = (8, 8)
    elif model_name == "multitask-ssast-frame-base-400":
        filename = "test_official_ssast_frame.pth"
        stride = (n_bins, 1)
    else:
        raise ValueError("Invalid model name is given.")

    model = SelfSupervisedAudioSpectrogramTransformer.build_from_pretrained(
        model_name,
        stride=stride,
        n_bins=n_bins,
        n_frames=n_frames,
        aggregator=aggregator,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        url = f"https://github.com/tky823/Audyn/releases/download/v0.0.1.dev3/{filename}"
        path = os.path.join(temp_dir, filename)
        download_file_from_github_release(url, path)

        data = torch.load(path)
        input = data["input"]
        expected_output = data["output"]

    model.eval()

    with torch.no_grad():
        output = model(input)

    allclose(output, expected_output, atol=1e-5)


@pytest.mark.parametrize("sample_wise", [True, False])
def test_ssast_multi_task_mpm(sample_wise: bool) -> None:
    torch.manual_seed(0)

    d_model = 8
    n_bins, n_frames = 8, 30
    kernel_size = (n_bins, 2)
    insert_cls_token, insert_dist_token = True, True
    batch_size = 4

    num_masks = 10
    min_cluster, max_cluster = 2, 3

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
    masker = FastMasker(
        d_model,
        num_masks=num_masks,
        min_cluster=min_cluster,
        max_cluster=max_cluster,
        sample_wise=sample_wise,
    )
    encoder_layer = nn.TransformerEncoderLayer(
        d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True
    )
    norm = nn.LayerNorm(d_model)
    transformer = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_layers,
        norm=norm,
    )
    reconstructor = MLP(d_model, kernel_size[0] * kernel_size[1])
    classifier = MLP(d_model, kernel_size[0] * kernel_size[1])
    model = MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel(
        patch_embedding,
        masker,
        transformer,
        reconstructor=reconstructor,
        classifier=classifier,
    )

    input = torch.randn((batch_size, n_bins, n_frames))
    reconstruction, classification = model(input)
    reconstruction_output, reconstruction_target, reconstruction_length = reconstruction
    classification_output, classification_target, classification_length = classification

    assert reconstruction_output.size(0) == input.size(0)
    assert reconstruction_output.size(0) == classification_output.size(0)
    assert reconstruction_output.size(-1) == classification_output.size(-1)
    assert reconstruction_target.size(0) == classification_target.size(0)
    assert reconstruction_target.size(-1) == classification_target.size(-1)
    assert reconstruction_length.size() == classification_length.size()


def test_ssast() -> None:
    torch.manual_seed(0)

    d_model, out_channels = 8, 10
    n_bins, n_frames = 8, 30
    kernel_size = (n_bins, 2)
    batch_size = 4

    nhead = 2
    dim_feedforward = 5
    num_layers = 2

    patch_embedding = PositionalPatchEmbedding(
        d_model,
        kernel_size=kernel_size,
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
    aggregator = AverageAggregator()
    head = MLPHead(d_model, out_channels)

    model = SelfSupervisedAudioSpectrogramTransformer(
        patch_embedding, transformer, aggregator=aggregator, head=head
    )

    input = torch.randn((batch_size, n_bins, n_frames))
    output = model(input)

    assert output.size() == (batch_size, out_channels)


@pytest.mark.parametrize("sample_wise", [True, False])
def test_ssast_masker(sample_wise: bool) -> None:
    torch.manual_seed(0)

    d_model = 8
    batch_size = 4

    num_masks = 50
    min_cluster, max_cluster = 3, 6

    model = Masker(
        d_model,
        num_masks=num_masks,
        min_cluster=min_cluster,
        max_cluster=max_cluster,
        sample_wise=sample_wise,
    )

    # patch-based SSAST-like inputs w/o padding_mask
    height, width = 10, 16
    num_paddings = 12
    input = torch.randn((batch_size, d_model, height, width))
    output, masking_mask = model(input)

    assert output.size() == input.size()
    assert masking_mask.size(0) == input.size(0)
    assert masking_mask.size()[1:] == input.size()[2:]
    assert torch.all(masking_mask.sum(dim=(-2, -1)) == num_masks)

    # patch-based SSAST-like inputs w/ padding_mask
    if sample_wise:
        padding_indices = torch.randperm(batch_size * height * width, dtype=torch.long)
        padding_indices = padding_indices[: batch_size * num_paddings]
        padding_mask = torch.zeros((batch_size * height * width,), dtype=torch.bool)
        padding_mask.scatter_(0, padding_indices, True)
        padding_mask = padding_mask.view(batch_size, height, width)
    else:
        padding_indices = torch.randperm(height * width, dtype=torch.long)
        padding_indices = padding_indices[:num_paddings]
        padding_mask = torch.zeros((height * width,), dtype=torch.bool)
        padding_mask.scatter_(0, padding_indices, True)
        padding_mask = padding_mask.view(height, width)
        padding_mask = padding_mask.expand((batch_size, -1, -1))

    output, masking_mask = model(input, padding_mask=padding_mask)

    assert output.size() == input.size()
    assert masking_mask.size(0) == input.size(0)
    assert masking_mask.size()[1:] == input.size()[2:]
    assert torch.all(masking_mask.sum(dim=(-2, -1)) == num_masks)

    masking_and_padding = torch.logical_and(masking_mask, padding_mask)
    assert not torch.any(masking_and_padding)

    # frame-based SSAST-like inputs w/o padding_mask
    height, width = 1, 100
    num_paddings = 10
    input = torch.randn((batch_size, d_model, height, width))
    output, masking_mask = model(input)

    assert output.size() == input.size()
    assert masking_mask.size(0) == input.size(0)
    assert masking_mask.size()[1:] == input.size()[2:]

    # frame-based SSAST-like inputs w/ padding_mask
    if sample_wise:
        padding_indices = torch.randperm(batch_size * height * width, dtype=torch.long)
        padding_indices = padding_indices[: batch_size * num_paddings]
        padding_mask = torch.zeros((batch_size * height * width,), dtype=torch.bool)
        padding_mask.scatter_(0, padding_indices, True)
        padding_mask = padding_mask.view(batch_size, height, width)
    else:
        padding_indices = torch.randperm(height * width, dtype=torch.long)
        padding_indices = padding_indices[:num_paddings]
        padding_mask = torch.zeros((height * width,), dtype=torch.bool)
        padding_mask.scatter_(0, padding_indices, True)
        padding_mask = padding_mask.view(height, width)
        padding_mask = padding_mask.expand((batch_size, -1, -1))

    output, masking_mask = model(input, padding_mask=padding_mask)

    assert output.size() == input.size()
    assert masking_mask.size(0) == input.size(0)
    assert masking_mask.size()[1:] == input.size()[2:]

    masking_and_padding = torch.logical_and(masking_mask, padding_mask)
    assert not torch.any(masking_and_padding)


@pytest.mark.parametrize("sample_wise", [True, False])
def test_ssast_fast_masker(sample_wise: bool) -> None:
    torch.manual_seed(0)

    d_model = 8
    height, width = 10, 16
    num_paddings = 12
    batch_size = 4

    num_masks = 50
    min_cluster, max_cluster = 3, 6

    model = FastMasker(
        d_model,
        num_masks=num_masks,
        min_cluster=min_cluster,
        max_cluster=max_cluster,
        sample_wise=sample_wise,
    )

    input = torch.randn((batch_size, d_model, height, width))

    # w/o padding_mask
    output, masking_mask = model(input)

    assert output.size() == input.size()
    assert masking_mask.size(0) == input.size(0)
    assert masking_mask.size()[1:] == input.size()[2:]

    # w/ padding_mask
    if sample_wise:
        padding_indices = torch.randperm(batch_size * height * width, dtype=torch.long)
        padding_indices = padding_indices[: batch_size * num_paddings]
        padding_mask = torch.zeros((batch_size * height * width,), dtype=torch.bool)
        padding_mask.scatter_(0, padding_indices, True)
        padding_mask = padding_mask.view(batch_size, height, width)
    else:
        padding_indices = torch.randperm(height * width, dtype=torch.long)
        padding_indices = padding_indices[:num_paddings]
        padding_mask = torch.zeros((height * width,), dtype=torch.bool)
        padding_mask.scatter_(0, padding_indices, True)
        padding_mask = padding_mask.view(height, width)
        padding_mask = padding_mask.expand((batch_size, -1, -1))

    output, masking_mask = model(input, padding_mask=padding_mask)

    assert output.size() == input.size()
    assert masking_mask.size(0) == input.size(0)
    assert masking_mask.size()[1:] == input.size()[2:]

    masking_and_padding = torch.logical_and(masking_mask, padding_mask)
    assert not torch.any(masking_and_padding)
