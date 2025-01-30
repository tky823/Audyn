import torch

from audyn.models.bsrnn import BandSplitRNN
from audyn.modules.bsrnn import (
    BandMergeModule,
    BandSplitModule,
    BandSplitRNNBackbone,
    MultiChannelBandMergeModule,
    MultiChannelBandSplitModule,
)


def test_bsrnn() -> None:
    torch.manual_seed(0)

    batch_size = 4
    bins = [15, 11, 7]
    n_frames = 128
    num_features, hidden_channels = 8, 6
    shape = (batch_size, sum(bins), n_frames)
    num_blocks = 3

    bandsplit = BandSplitModule(bins, num_features)
    bandmerge = BandMergeModule(bins, num_features)
    backbone = BandSplitRNNBackbone(num_features, hidden_channels, num_blocks=num_blocks)
    model = BandSplitRNN(bandsplit, bandmerge, backbone)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    assert output.size() == input.size()


def test_multichannel_bsrnn() -> None:
    torch.manual_seed(0)

    batch_size = 4
    in_channels = 3
    bins = [15, 11, 5, 2]
    n_frames = 128
    num_features, hidden_channels = 8, 6
    shape = (batch_size, in_channels, sum(bins), n_frames)
    num_blocks = 3

    bandsplit = MultiChannelBandSplitModule(in_channels, bins, num_features)
    bandmerge = MultiChannelBandMergeModule(in_channels, bins, num_features)
    backbone = BandSplitRNNBackbone(num_features, hidden_channels, num_blocks=num_blocks)
    model = BandSplitRNN(bandsplit, bandmerge, backbone)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    assert output.size() == input.size()
