import copy

import torch
import torch.nn as nn

from audyn.modules.vqvae import VectorQuantizer


def test_vector_quantizer():
    """Ensure gradient of input is same as that of output."""
    torch.manual_seed(0)

    batch_size = 2
    height, width = 3, 3
    codebook_size, embedding_dim = 5, 4

    vector_quantizer = VectorQuantizer(codebook_size, embedding_dim=embedding_dim)
    embedding = nn.Embedding.from_pretrained(
        vector_quantizer.codebook.weight.data.detach(),
        freeze=True,
    )

    input = torch.randn((batch_size, embedding_dim, height, width), requires_grad=True)
    quantized_by_quantizer, indices = vector_quantizer(input)
    quantized_by_embedding = embedding(indices)
    quantized_by_embedding = quantized_by_embedding.permute(0, 3, 1, 2)

    # confirm gradient of input
    assert torch.allclose(quantized_by_quantizer, quantized_by_embedding)

    # k-means clustering initalization
    kmeans_iteration = 100

    vector_quantizer = VectorQuantizer(
        codebook_size,
        embedding_dim=embedding_dim,
        init_by_kmeans=kmeans_iteration,
    )

    _ = vector_quantizer(input)
    _, indices_before_save = vector_quantizer(input)
    state_dict = copy.copy(vector_quantizer.state_dict())
    vector_quantizer.load_state_dict(state_dict)

    _, indices_after_save = vector_quantizer(input)

    assert torch.equal(indices_before_save, indices_after_save)
