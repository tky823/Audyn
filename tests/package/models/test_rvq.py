import torch
from dummy.modules.vqvae import Decoder, Encoder

from audyn.models.rvqvae import RVQVAE


def test_rvqvae():
    torch.manual_seed(0)

    batch_size = 4
    codebook_size = 5
    in_channels, hidden_channels = 1, 16
    stride = 2
    num_layers, num_rvq_layers = 2, 3
    height, width = 32, 32
    latent_size = (height // (stride**num_layers), width // (stride**num_layers))

    encoder = Encoder(in_channels, hidden_channels, stride=stride, num_layers=num_layers)
    decoder = Decoder(in_channels, hidden_channels, stride=stride, num_layers=num_layers)
    model = RVQVAE(
        encoder,
        decoder,
        codebook_size=codebook_size,
        embedding_dim=hidden_channels,
        num_layers=num_rvq_layers,
        dropout=False,
    )

    input = torch.randn((batch_size, in_channels, height, width))
    reconstructed, encoded, hierarchical_quantized, indices = model(input)
    quantized = hierarchical_quantized.sum(dim=1)

    assert reconstructed.size() == input.size()
    assert encoded.size(0) == hierarchical_quantized.size(0)
    assert encoded.size()[1:] == hierarchical_quantized.size()[2:]
    assert hierarchical_quantized.size(0) == indices.size(0)
    assert hierarchical_quantized.size(1) == num_rvq_layers
    assert hierarchical_quantized.size(2) == hidden_channels
    assert hierarchical_quantized.size()[3:] == latent_size
    assert indices.size()[2:] == latent_size

    output_by_hierarchical_quantized = model.inference(hierarchical_quantized, layer_wise=True)
    output_by_quantized = model.inference(quantized, layer_wise=False)
    output_by_indices = model.inference(indices, layer_wise=True)

    assert torch.allclose(output_by_indices, output_by_hierarchical_quantized)
    assert torch.allclose(output_by_indices, output_by_quantized)

    hierarchical_quantized, indices = model.sample(input)

    assert hierarchical_quantized.size(0) == indices.size(0)
    assert hierarchical_quantized.size(1) == num_rvq_layers
    assert hierarchical_quantized.size(2) == hidden_channels
    assert hierarchical_quantized.size()[3:] == latent_size
    assert indices.size()[2:] == latent_size

    hierarchical_quantized, indices = model.rsample(input)

    assert hierarchical_quantized.size(0) == indices.size(0)
    assert hierarchical_quantized.size(1) == num_rvq_layers
    assert hierarchical_quantized.size(2) == hidden_channels
    assert hierarchical_quantized.size()[3:] == latent_size
    assert indices.size()[2:] == latent_size
