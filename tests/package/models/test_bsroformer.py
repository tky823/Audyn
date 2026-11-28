import os
from pathlib import Path

import hydra
import pytest
import torch
import torch.nn as nn
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir
from omegaconf import OmegaConf

from audyn.models.bsroformer import (
    AudioSeparatorCompatibleBandSplitRoFormer,
    BandSplitRoFormer,
    MultiSourceMultiChannelBandSplitRoFormer,
)
from audyn.modules.bsrnn import (
    AudioSeparatorCompatibleBandMergeModule,
    AudioSeparatorCompatibleBandSplitModule,
    BandMergeModule,
    BandSplitModule,
    MultiChannelBandMergeModule,
    MultiChannelBandSplitModule,
    MultiSourceMultiChannelBandMergeModule,
)
from audyn.modules.bsroformer import (
    AudioSeparatorCompatibleBandSplitRoFormerBackbone,
    BandSplitRoFormerBackbone,
)
from audyn.utils._github import download_file_from_github_release


@pytest.mark.slow
def test_large_bsroformer() -> None:
    batch_size = 4
    n_frames = 64

    # Band-split RoFormer
    in_channels = 2
    version = "default"
    model = BandSplitRoFormer.build_from_config(in_channels, version=version)
    n_bins = sum(model.bandsplit.bins)

    num_paramters = 0

    for p in model.parameters():
        if p.requires_grad:
            num_paramters += p.numel()

    assert num_paramters == 93672756

    shape = (batch_size, in_channels, n_bins, n_frames)
    input = torch.randn(shape) + 1j * torch.randn(shape)

    with torch.no_grad():
        output = model(input)

    assert output.size() == input.size()

    # BandIt-like
    num_sources = 3
    in_channels = 1
    version = "music-scale"
    model = MultiSourceMultiChannelBandSplitRoFormer.build_from_config(
        num_sources, in_channels, version=version
    )
    _, n_bins = model.bandsplit.bins[-1]

    shape = (batch_size, in_channels, n_bins, n_frames)
    input = torch.randn(shape) + 1j * torch.randn(shape)

    with torch.no_grad():
        output = model(input)

    assert output.size() == (batch_size, num_sources, in_channels, n_bins, n_frames)


def test_bsroformer() -> None:
    torch.manual_seed(0)

    batch_size = 4
    bins = [15, 11, 7]
    n_frames = 64
    num_features, head_channels, hidden_channels = 8, 4, 6
    shape = (batch_size, sum(bins), n_frames)
    num_heads = 2
    num_blocks = 3

    bandsplit = BandSplitModule(bins, num_features)
    bandmerge = BandMergeModule(bins, num_features)
    backbone = BandSplitRoFormerBackbone(
        num_features, num_heads, head_channels, hidden_channels, num_blocks=num_blocks
    )
    model = BandSplitRoFormer(bandsplit, bandmerge, backbone)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    assert output.size() == input.size()

    bins = [[0, 20], [15, 28], [24, 33]]

    bandsplit = BandSplitModule(bins, num_features)
    bandmerge = BandMergeModule(bins, num_features)
    backbone = BandSplitRoFormerBackbone(
        num_features, num_heads, head_channels, hidden_channels, num_blocks=num_blocks
    )
    model = BandSplitRoFormer(bandsplit, bandmerge, backbone)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    assert output.size() == input.size()


def test_multichannel_bsroformer() -> None:
    torch.manual_seed(0)

    batch_size = 4
    in_channels = 3
    bins = [15, 11, 5, 2]
    n_frames = 64
    num_features, head_channels, hidden_channels = 8, 4, 6
    shape = (batch_size, in_channels, sum(bins), n_frames)
    num_heads = 2
    num_blocks = 3

    bandsplit = MultiChannelBandSplitModule(in_channels, bins, num_features)
    bandmerge = MultiChannelBandMergeModule(in_channels, bins, num_features)
    backbone = BandSplitRoFormerBackbone(
        num_features, num_heads, head_channels, hidden_channels, num_blocks=num_blocks
    )
    model = BandSplitRoFormer(bandsplit, bandmerge, backbone)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    assert output.size() == input.size()

    bins = [[0, 18], [12, 28], [24, 31], [30, 33]]

    bandsplit = MultiChannelBandSplitModule(in_channels, bins, num_features)
    bandmerge = MultiChannelBandMergeModule(in_channels, bins, num_features)
    backbone = BandSplitRoFormerBackbone(
        num_features, num_heads, head_channels, hidden_channels, num_blocks=num_blocks
    )
    model = BandSplitRoFormer(bandsplit, bandmerge, backbone)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    assert output.size() == input.size()


def test_multisource_multichannel_bsroformer() -> None:
    torch.manual_seed(0)

    batch_size = 4
    in_channels = 1
    bins = [15, 11, 5, 2]
    n_frames = 64
    num_features, head_channels, hidden_channels = 8, 4, 6
    shape = (batch_size, in_channels, sum(bins), n_frames)
    num_sources = 3
    num_heads = 2
    num_blocks = 2

    bandsplit = MultiChannelBandSplitModule(in_channels, bins, num_features)
    bandmerge = MultiSourceMultiChannelBandMergeModule(
        num_sources, in_channels, bins, num_features
    )
    backbone = BandSplitRoFormerBackbone(
        num_features, num_heads, head_channels, hidden_channels, num_blocks=num_blocks
    )
    model = MultiSourceMultiChannelBandSplitRoFormer(bandsplit, bandmerge, backbone)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    assert output.size() == (batch_size, num_sources, in_channels, sum(bins), n_frames)

    bins = [[0, 18], [12, 28], [24, 31], [30, 33]]

    bandsplit = MultiChannelBandSplitModule(in_channels, bins, num_features)
    bandmerge = MultiSourceMultiChannelBandMergeModule(
        num_sources, in_channels, bins, num_features
    )
    backbone = BandSplitRoFormerBackbone(
        num_features, num_heads, head_channels, hidden_channels, num_blocks=num_blocks
    )
    model = MultiSourceMultiChannelBandSplitRoFormer(bandsplit, bandmerge, backbone)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    output = model(input)

    _, n_bins = bins[-1]

    assert output.size() == (batch_size, num_sources, in_channels, n_bins, n_frames)


def test_audio_separator_compatible_bsroformer_build_from_config() -> None:
    torch.manual_seed(0)

    in_channels = 1

    _ = AudioSeparatorCompatibleBandSplitRoFormer.build_from_config(in_channels)


@pytest.mark.slow
def test_official_audio_separator_compatible_bsroformer() -> None:
    torch.manual_seed(0)

    url = "https://github.com/tky823/Audyn/releases/download/v0.4.0/test_audio-separator_melbsroformer.pth"  # noqa: E501

    filename = os.path.basename(url)
    path = os.path.join(audyn_test_cache_dir, filename)
    download_file_from_github_release(url, path)

    model = AudioSeparatorCompatibleBandSplitRoFormer.build_from_pretrained(
        "audio-separator_melbsroformer"
    )

    data = torch.load(path)
    expected_spectrogram = data["output"]["transform"]
    expected_bandsplit_output = data["output"]["bandsplit"]
    expected_backbone_output = data["output"]["backbone"]
    expected_bandmerge_output = data["output"]["bandmerge"]

    model.eval()

    expected_spectrogram = expected_spectrogram.unsqueeze(dim=0)
    expected_bandsplit_output = expected_bandsplit_output.unsqueeze(dim=0)
    expected_backbone_output = expected_backbone_output.unsqueeze(dim=0)

    with torch.no_grad():
        bandsplit_output = model.bandsplit(expected_spectrogram)
        backbone_output = model.backbone(expected_bandsplit_output)
        bandmerge_output = model.bandmerge(expected_backbone_output)

    allclose(bandsplit_output, expected_bandsplit_output, atol=1e-4)
    allclose(backbone_output, expected_backbone_output, atol=1e-4)
    allclose(bandmerge_output, expected_bandmerge_output, atol=1e-4)


def test_audio_separator_compatible_bsroformer_without_torch_rmsnorm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(nn, "RMSNorm", raising=False)

    bins = [2, 2]
    bandsplit = AudioSeparatorCompatibleBandSplitModule(1, bins, 8, norm="rms")
    bandmerge = AudioSeparatorCompatibleBandMergeModule(1, bins, 8, hidden_channels=16, norm="rms")
    backbone = AudioSeparatorCompatibleBandSplitRoFormerBackbone(
        8, 2, 4, 16, num_blocks=1, norm="rms", dropout=0, bias=False
    )
    model = AudioSeparatorCompatibleBandSplitRoFormer(bandsplit, bandmerge, backbone)

    input = torch.randn(1, 1, 4, 3, dtype=torch.complex64)
    output = model(input)

    assert output.size() == input.size()


def test_audio_separator_compatible_bsroformer_loads_local_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def build_small_model() -> AudioSeparatorCompatibleBandSplitRoFormer:
        bins = [[0, 2], [1, 3]]
        return AudioSeparatorCompatibleBandSplitRoFormer(
            AudioSeparatorCompatibleBandSplitModule(1, bins, 8, norm="rms"),
            AudioSeparatorCompatibleBandMergeModule(1, bins, 8, hidden_channels=12),
            AudioSeparatorCompatibleBandSplitRoFormerBackbone(
                8, 2, 4, 12, num_blocks=1, norm="rms", dropout=0, bias=False
            ),
        )

    torch.manual_seed(0)
    expected = build_small_model()
    checkpoint = tmp_path / "model.pth"
    torch.save(
        {
            "resolved_config": OmegaConf.to_container(
                OmegaConf.create({"model": {"_target_": "unused.for.mock"}})
            ),
            "model": expected.state_dict(),
        },
        checkpoint,
    )
    monkeypatch.setattr(hydra.utils, "instantiate", lambda config: build_small_model())

    actual = AudioSeparatorCompatibleBandSplitRoFormer.build_from_pretrained(str(checkpoint))
    input = torch.randn(1, 1, 3, 4, dtype=torch.complex64)
    expected.eval()
    actual.eval()

    with torch.no_grad():
        torch.testing.assert_close(actual(input), expected(input))


def test_audio_separator_compatible_bsroformer_unknown_checkpoint() -> None:
    with pytest.raises(ValueError, match="is not found"):
        AudioSeparatorCompatibleBandSplitRoFormer.build_from_pretrained("unknown-model")
