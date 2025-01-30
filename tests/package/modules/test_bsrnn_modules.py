import torch

from audyn.modules.bsrnn import (
    BandMergeBlock,
    BandMergeModule,
    BandSplitBlock,
    BandSplitModule,
    BandSplitRNNBackbone,
    BandSplitRNNBlock,
    MultiChannelBandMergeModule,
    MultiChannelBandSplitModule,
)


def test_bsrnn_backbone() -> None:
    torch.manual_seed(0)

    batch_size, n_bands, n_frames = 4, 3, 128
    num_features, hidden_channels = 6, 5
    num_blocks = 6

    model = BandSplitRNNBackbone(num_features, hidden_channels, num_blocks=num_blocks)
    input = torch.randn((batch_size, num_features, n_bands, n_frames))

    output = model(input)

    assert output.size() == (batch_size, num_features, n_bands, n_frames)


def test_bsrnn_block() -> None:
    torch.manual_seed(0)

    batch_size, n_bands, n_frames = 4, 3, 128
    num_features, hidden_channels = 6, 5

    model = BandSplitRNNBlock(num_features, hidden_channels)
    input = torch.randn((batch_size, num_features, n_bands, n_frames))

    output = model(input)

    assert output.size() == (batch_size, num_features, n_bands, n_frames)


def test_bsrnn_band_split_block() -> None:
    torch.manual_seed(0)

    batch_size, n_bins, n_frames = 4, 10, 128
    embed_dim = 8
    shape = (batch_size, n_bins, n_frames)

    model = BandSplitBlock(n_bins, embed_dim)
    input = torch.randn(shape) + 1j * torch.randn(shape)

    output = model(input)

    assert output.size() == (batch_size, embed_dim, n_frames)


def test_bsrnn_band_merge_block() -> None:
    torch.manual_seed(0)

    batch_size, n_bins, n_frames = 4, 10, 128
    embed_dim = 8

    model = BandMergeBlock(n_bins, embed_dim)
    input = torch.randn((batch_size, embed_dim, n_frames))

    output = model(input)

    assert output.size() == (batch_size, n_bins, n_frames)


def test_bsrnn_bandsplit_module() -> None:
    torch.manual_seed(0)

    batch_size = 4
    bins, n_frames = [10, 8, 5], 128
    embed_dim = 8
    shape = (batch_size, sum(bins), n_frames)

    model = BandSplitModule(bins, embed_dim)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    assert output.size() == (batch_size, embed_dim, len(bins), n_frames)


def test_bsrnn_bandmerge_module() -> None:
    torch.manual_seed(0)

    batch_size = 4
    bins, n_frames = [10, 8, 5], 128
    embed_dim = 8

    model = BandMergeModule(bins, embed_dim)
    input = torch.randn((batch_size, embed_dim, len(bins), n_frames))
    output = model(input)

    assert output.size() == (batch_size, sum(bins), n_frames)


def test_bsrnn_multichannel_bandsplit_module() -> None:
    torch.manual_seed(0)

    batch_size = 4
    in_channels = 3
    bins, n_frames = [10, 8, 5], 128
    embed_dim = 8
    shape = (batch_size, in_channels, sum(bins), n_frames)

    model = MultiChannelBandSplitModule(in_channels, bins, embed_dim)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    assert output.size() == (batch_size, embed_dim, len(bins), n_frames)


def test_bsrnn_multichannel_bandmerge_module() -> None:
    torch.manual_seed(0)

    batch_size = 4
    out_channels = 3
    bins, n_frames = [10, 8, 5], 128
    embed_dim = 8

    model = MultiChannelBandMergeModule(out_channels, bins, embed_dim)
    input = torch.randn((batch_size, embed_dim, len(bins), n_frames))
    output = model(input)

    assert output.size() == (batch_size, out_channels, sum(bins), n_frames)
