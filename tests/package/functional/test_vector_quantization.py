import torch
from audyn_test import allclose

from audyn.functional.vector_quantization import (
    quantize_gumbel_vector,
    quantize_residual_vector,
)


def test_quantize_gumbel_vector_1d() -> None:
    torch.manual_seed(0)

    batch_size = 4
    codebook_size, embedding_dim = 10, 5
    length = 3

    input = torch.randn((batch_size, codebook_size, length))
    weight = torch.randn((codebook_size, embedding_dim))
    quantized, indices = quantize_gumbel_vector(input, weight)

    assert quantized.size() == (batch_size, embedding_dim, length)
    assert indices.size() == (batch_size, length)


def test_quantize_gumbel_vector_2d() -> None:
    torch.manual_seed(0)

    batch_size = 4
    codebook_size, embedding_dim = 10, 5
    height, width = 2, 3

    input = torch.randn((batch_size, codebook_size, height, width))
    weight = torch.randn((codebook_size, embedding_dim))
    quantized, indices = quantize_gumbel_vector(input, weight)

    assert quantized.size() == (batch_size, embedding_dim, height, width)
    assert indices.size() == (batch_size, height, width)


def test_quantize_residual_vector_1d() -> None:
    torch.manual_seed(0)

    batch_size = 4
    num_layers = 6
    codebook_size, embedding_dim = 10, 5
    length = 3

    input = torch.randn((batch_size, embedding_dim, length))

    # weight is tensor
    weight = torch.randn((num_layers, codebook_size, embedding_dim))
    quantized, residual, indices = quantize_residual_vector(input, weight)

    assert quantized.size() == (batch_size, num_layers, embedding_dim, length)
    assert residual.size() == (batch_size, num_layers, embedding_dim, length)
    assert indices.size() == (batch_size, num_layers, length)

    quantized = quantized.transpose(1, 0)
    residual = residual.transpose(1, 0)

    for layer_idx in range(num_layers):
        _residual = residual[layer_idx]

        if layer_idx == 0:
            allclose(_residual, input)
        else:
            _quantized = torch.sum(quantized[:layer_idx], dim=0)
            allclose(_quantized + _residual, input)

    # weight is list of tensors
    weight = []

    for layer_idx in range(num_layers):
        _weight = torch.randn((codebook_size + layer_idx, embedding_dim))
        weight.append(_weight)

    quantized, residual, indices = quantize_residual_vector(input, weight)

    assert quantized.size() == (batch_size, num_layers, embedding_dim, length)
    assert residual.size() == (batch_size, num_layers, embedding_dim, length)
    assert indices.size() == (batch_size, num_layers, length)

    quantized = quantized.transpose(1, 0)
    residual = residual.transpose(1, 0)

    for layer_idx in range(num_layers):
        _residual = residual[layer_idx]

        if layer_idx == 0:
            allclose(_residual, input)
        else:
            _quantized = torch.sum(quantized[:layer_idx], dim=0)
            allclose(_quantized + _residual, input)


def test_quantize_residual_vector_2d() -> None:
    torch.manual_seed(0)

    batch_size = 4
    num_layers = 6
    codebook_size, embedding_dim = 10, 5
    height, width = 2, 3

    input = torch.randn((batch_size, embedding_dim, height, width))

    # weight is tensor
    weight = torch.randn((num_layers, codebook_size, embedding_dim))
    quantized, residual, indices = quantize_residual_vector(input, weight)

    assert quantized.size() == (batch_size, num_layers, embedding_dim, height, width)
    assert residual.size() == (batch_size, num_layers, embedding_dim, height, width)
    assert indices.size() == (batch_size, num_layers, height, width)

    quantized = quantized.transpose(1, 0)
    residual = residual.transpose(1, 0)

    for layer_idx in range(num_layers):
        _residual = residual[layer_idx]

        if layer_idx == 0:
            allclose(_residual, input)
        else:
            _quantized = torch.sum(quantized[:layer_idx], dim=0)
            allclose(_quantized + _residual, input, atol=1e-7)

    # weight is list of tensors
    weight = []

    for layer_idx in range(num_layers):
        _weight = torch.randn((codebook_size + layer_idx, embedding_dim))
        weight.append(_weight)

    quantized, residual, indices = quantize_residual_vector(input, weight)

    assert quantized.size() == (batch_size, num_layers, embedding_dim, height, width)
    assert residual.size() == (batch_size, num_layers, embedding_dim, height, width)
    assert indices.size() == (batch_size, num_layers, height, width)

    quantized = quantized.transpose(1, 0)
    residual = residual.transpose(1, 0)

    for layer_idx in range(num_layers):
        _residual = residual[layer_idx]

        if layer_idx == 0:
            allclose(_residual, input)
        else:
            _quantized = torch.sum(quantized[:layer_idx], dim=0)
            allclose(_quantized + _residual, input, atol=1e-7)
