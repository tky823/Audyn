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
