import pytest
import torch

from audyn.models.bsrnn import BandSplitRNN, MultiSourceMultiChannelBandSplitRNN
from audyn.modules.bsrnn import (
    BandMergeModule,
    BandSplitModule,
    BandSplitRNNBackbone,
    MultiChannelBandMergeModule,
    MultiChannelBandSplitModule,
    MultiSourceMultiChannelBandMergeModule,
)


@pytest.mark.slow
def test_official_bsrnn() -> None:
    batch_size = 4
    n_frames = 128

    # BandSplitRNN
    in_channels = 2
    version = "v7"
    model = BandSplitRNN.build_from_config(in_channels, version=version)
    n_bins = sum(model.bandsplit.bins)

    shape = (batch_size, in_channels, n_bins, n_frames)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    assert output.size() == input.size()

    # BandIt
    num_sources = 3
    in_channels = 1
    version = "music-scale"
    model = MultiSourceMultiChannelBandSplitRNN.build_from_config(
        num_sources, in_channels, version=version
    )
    _, n_bins = model.bandsplit.bins[-1]

    shape = (batch_size, in_channels, n_bins, n_frames)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    assert output.size() == (batch_size, num_sources, in_channels, n_bins, n_frames)


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

    bins = [[0, 20], [15, 28], [24, 33]]

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

    bins = [[0, 18], [12, 28], [24, 31], [30, 33]]

    bandsplit = MultiChannelBandSplitModule(in_channels, bins, num_features)
    bandmerge = MultiChannelBandMergeModule(in_channels, bins, num_features)
    backbone = BandSplitRNNBackbone(num_features, hidden_channels, num_blocks=num_blocks)
    model = BandSplitRNN(bandsplit, bandmerge, backbone)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    assert output.size() == input.size()


def test_multisource_multichannel_bsrnn() -> None:
    torch.manual_seed(0)

    batch_size = 4
    in_channels = 1
    bins = [15, 11, 5, 2]
    n_frames = 128
    num_features, hidden_channels = 8, 6
    shape = (batch_size, in_channels, sum(bins), n_frames)
    num_sources = 3
    num_blocks = 2

    bandsplit = MultiChannelBandSplitModule(in_channels, bins, num_features)
    bandmerge = MultiSourceMultiChannelBandMergeModule(
        num_sources, in_channels, bins, num_features
    )
    backbone = BandSplitRNNBackbone(num_features, hidden_channels, num_blocks=num_blocks)
    model = MultiSourceMultiChannelBandSplitRNN(bandsplit, bandmerge, backbone)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    assert output.size() == (batch_size, num_sources, in_channels, sum(bins), n_frames)

    bins = [[0, 18], [12, 28], [24, 31], [30, 33]]

    bandsplit = MultiChannelBandSplitModule(in_channels, bins, num_features)
    bandmerge = MultiSourceMultiChannelBandMergeModule(
        num_sources, in_channels, bins, num_features
    )
    backbone = BandSplitRNNBackbone(num_features, hidden_channels, num_blocks=num_blocks)
    model = MultiSourceMultiChannelBandSplitRNN(bandsplit, bandmerge, backbone)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    _, n_bins = bins[-1]

    assert output.size() == (batch_size, num_sources, in_channels, n_bins, n_frames)
