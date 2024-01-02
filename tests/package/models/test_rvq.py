import copy

import torch
from dummy.modules.vqvae import Decoder, Encoder

from audyn.models.rvqvae import RVQVAE


def test_rvqvae():
    torch.manual_seed(0)

    batch_size = 4
    codebook_size = 5
    in_channels, hidden_channels = 1, 16
    stride = 2
    num_layers, num_stages = 2, 3
    height, width = 32, 32
    latent_size = (height // (stride**num_layers), width // (stride**num_layers))

    encoder = Encoder(in_channels, hidden_channels, stride=stride, num_layers=num_layers)
    decoder = Decoder(in_channels, hidden_channels, stride=stride, num_layers=num_layers)
    model = RVQVAE(
        encoder,
        decoder,
        codebook_size=codebook_size,
        embedding_dim=hidden_channels,
        num_stages=num_stages,
        dropout=False,
    )

    input = torch.randn((batch_size, in_channels, height, width))
    reconstructed, encoded, hierarchical_quantized, indices = model(input)
    quantized = hierarchical_quantized.sum(dim=1)

    assert reconstructed.size() == input.size()
    assert encoded.size(0) == hierarchical_quantized.size(0)
    assert encoded.size()[1:] == hierarchical_quantized.size()[2:]
    assert hierarchical_quantized.size(0) == indices.size(0)
    assert hierarchical_quantized.size(1) == num_stages
    assert hierarchical_quantized.size(2) == hidden_channels
    assert hierarchical_quantized.size()[3:] == latent_size
    assert indices.size()[2:] == latent_size

    output_by_hierarchical_quantized = model.inference(hierarchical_quantized, stage_wise=True)
    output_by_quantized = model.inference(quantized, stage_wise=False)
    output_by_indices = model.inference(indices, stage_wise=True)

    assert torch.allclose(output_by_indices, output_by_hierarchical_quantized)
    assert torch.allclose(output_by_indices, output_by_quantized)

    hierarchical_quantized, indices = model.sample(input)

    assert hierarchical_quantized.size(0) == indices.size(0)
    assert hierarchical_quantized.size(1) == num_stages
    assert hierarchical_quantized.size(2) == hidden_channels
    assert hierarchical_quantized.size()[3:] == latent_size
    assert indices.size()[2:] == latent_size

    hierarchical_quantized, indices = model.rsample(input)

    assert hierarchical_quantized.size(0) == indices.size(0)
    assert hierarchical_quantized.size(1) == num_stages
    assert hierarchical_quantized.size(2) == hidden_channels
    assert hierarchical_quantized.size()[3:] == latent_size
    assert indices.size()[2:] == latent_size

    # k-means clustering initialization
    kmeans_initalization = 100

    encoder = Encoder(in_channels, hidden_channels, stride=stride, num_layers=num_layers)
    decoder = Decoder(in_channels, hidden_channels, stride=stride, num_layers=num_layers)
    model = RVQVAE(
        encoder,
        decoder,
        codebook_size=codebook_size,
        embedding_dim=hidden_channels,
        num_stages=num_stages,
        dropout=False,
        init_by_kmeans=kmeans_initalization,
    )

    input = torch.randn((batch_size, in_channels, height, width))
    _ = model(input)

    state_dict = copy.copy(model.state_dict())
    model.load_state_dict(state_dict)
