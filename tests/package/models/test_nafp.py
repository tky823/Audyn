import torch

from audyn.models.nafp import (
    ContrastiveNeuralAudioFingerprinter,
    NeuralAudioFingerprinter,
)
from audyn.modules.nafp import (
    NeuralAudioFingerprinterBackbone,
    NeuralAudioFingerprinterProjection,
)


def test_nafp() -> None:
    torch.manual_seed(0)

    in_channels, hidden_channels, embedding_dim = 1, 8, 16
    num_features = [
        embedding_dim,
        2 * embedding_dim,
        4 * embedding_dim,
        128,
    ]
    kernel_size, stride = 3, 2
    batch_size = 4
    n_bins, n_frames = 16, 4

    model = NeuralAudioFingerprinter(
        NeuralAudioFingerprinterBackbone(
            in_channels, num_features, kernel_size=kernel_size, stride=stride
        ),
        NeuralAudioFingerprinterProjection(num_features[-1], embedding_dim, hidden_channels),
    )

    input = torch.randn((batch_size, in_channels, n_bins, n_frames))
    output = model(input)

    assert output.size() == (batch_size, embedding_dim)

    model = ContrastiveNeuralAudioFingerprinter(
        NeuralAudioFingerprinterBackbone(in_channels, num_features, kernel_size=kernel_size),
        NeuralAudioFingerprinterProjection(num_features[-1], embedding_dim, hidden_channels),
    )

    input = torch.randn((batch_size, in_channels, n_bins, n_frames))
    other = torch.randn((batch_size, in_channels, n_bins, n_frames))

    output_one, output_other = model(input, other)

    assert output_one.size() == (batch_size, embedding_dim)
    assert output_other.size() == (batch_size, embedding_dim)
