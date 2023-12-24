import torch

from audyn.modules.rvq import ResidualVectorQuantizer


def test_residual_vector_quantizer() -> None:
    torch.manual_seed(0)

    batch_size = 4
    num_layers = 6
    codebook_size, embedding_dim = 10, 5
    length = 3

    input = torch.randn((batch_size, embedding_dim, length))

    rvq = ResidualVectorQuantizer(codebook_size, embedding_dim, num_layers=num_layers)
    quantized, indices = rvq(input)

    assert quantized.size() == (batch_size, num_layers, embedding_dim, length)
    assert indices.size() == (batch_size, num_layers, length)
