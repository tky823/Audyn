import pytest
import torch

from audyn.modules.passt import (
    DisentangledPositionalPatchEmbedding,
    StructuredPatchout,
    UnstructuredPatchout,
)


def test_passt_disentangled_positional_patch_embedding() -> None:
    torch.manual_seed(0)

    d_model = 8
    n_bins, n_frames = 8, 30
    kernel_size = (4, 4)
    stride = (2, 2)
    batch_size = 4

    model = DisentangledPositionalPatchEmbedding(
        d_model,
        kernel_size=kernel_size,
        stride=stride,
        n_bins=n_bins,
        n_frames=n_frames,
    )

    input = torch.randn((batch_size, n_bins, n_frames))
    output = model(input)

    Kh, Kw = kernel_size
    Sh, Sw = stride
    height = (n_bins - Kh) // Sh + 1
    width = (n_frames - Kw) // Sw + 1

    assert output.size() == (batch_size, height * width, d_model)


@pytest.mark.parametrize("sample_wise", [True, False])
def test_passt_unstructured_patchout(sample_wise: bool) -> None:
    torch.manual_seed(0)

    embedding_dim = 4
    height, width = 8, 20
    num_drops = 8

    batch_size = 4

    model = UnstructuredPatchout(
        num_drops=num_drops,
        sample_wise=sample_wise,
    )

    input = torch.randn((batch_size, embedding_dim, height, width))
    output, length = model(input)

    assert output.size(0) == batch_size
    assert output.size(-1) == embedding_dim
    assert length.size(0) == batch_size


@pytest.mark.parametrize("sample_wise", [True, False])
def test_passt_structured_patchout(sample_wise: bool) -> None:
    torch.manual_seed(0)

    embedding_dim = 4
    height, width = 8, 20
    num_frequency_drops, num_time_drops = 2, 4

    batch_size = 4

    model = StructuredPatchout(
        num_frequency_drops=num_frequency_drops,
        num_time_drops=num_time_drops,
        sample_wise=sample_wise,
    )

    input = torch.randn((batch_size, embedding_dim, height, width))
    output, length = model(input)

    assert output.size(0) == batch_size
    assert output.size(-1) == embedding_dim
    assert length.size(0) == batch_size
