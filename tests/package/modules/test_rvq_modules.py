import copy

import torch

from audyn.modules.rvq import ResidualVectorQuantizer


def test_residual_vector_quantizer() -> None:
    torch.manual_seed(0)

    batch_size = 4
    num_stages = 6
    codebook_size, embedding_dim = 10, 5
    length = 3

    input = torch.randn((batch_size, embedding_dim, length))

    rvq = ResidualVectorQuantizer(
        codebook_size,
        embedding_dim,
        num_stages=num_stages,
        dropout=False,
    )
    quantized, indices = rvq(input)

    assert quantized.size() == (batch_size, num_stages, embedding_dim, length)
    assert indices.size() == (batch_size, num_stages, length)

    # k-means clustering initalization
    kmeans_iteration = 100

    vector_quantizer = ResidualVectorQuantizer(
        codebook_size,
        embedding_dim,
        num_stages=num_stages,
        dropout=False,
        init_by_kmeans=kmeans_iteration,
    )

    _ = vector_quantizer(input)
    _, indices_before_save = vector_quantizer(input)
    state_dict = copy.copy(vector_quantizer.state_dict())
    vector_quantizer.load_state_dict(state_dict)

    _, indices_after_save = vector_quantizer(input)

    assert torch.equal(indices_before_save, indices_after_save)
