import torch
import torch.nn as nn

from audyn.models.ssast import (
    MLP,
    Masker,
    MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel,
    PositionalPatchEmbedding,
)


def test_ssast() -> None:
    torch.manual_seed(0)

    d_model = 8
    n_bins, n_frames = 8, 30
    kernel_size = (n_bins, 2)
    batch_size = 4

    num_masks = 10
    min_cluster, max_cluster = 2, 3

    nhead = 2
    dim_feedforward = 5

    patch_embedding = PositionalPatchEmbedding(
        d_model,
        kernel_size=kernel_size,
        n_bins=n_bins,
        n_frames=n_frames,
    )
    masker = Masker(
        d_model,
        num_masks=num_masks,
        min_cluster=min_cluster,
        max_cluster=max_cluster,
    )
    encoder_layer = nn.TransformerEncoderLayer(
        d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True
    )
    transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
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


def test_ssast_positional_patch_embedding() -> None:
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

    assert output.size() == (batch_size, d_model, height, width)


def test_ssast_masker() -> None:
    torch.manual_seed(0)

    d_model = 8
    height, width = 10, 30
    batch_size = 4

    num_masks = 10
    min_cluster, max_cluster = 2, 3

    model = Masker(d_model, num_masks=num_masks, min_cluster=min_cluster, max_cluster=max_cluster)

    input = torch.randn((batch_size, d_model, height, width))
    output, masking_mask = model(input)

    assert output.size() == input.size()
    assert masking_mask.size(0) == input.size(0)
    assert masking_mask.size()[1:] == input.size()[2:]
