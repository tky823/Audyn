import os

import torch
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir

from audyn.models.musicfm import MusicFM, MusicFMMaskedTokenModel
from audyn.utils._github import download_file_from_github_release


def test_musicfm() -> None:
    # regression test
    url = "https://github.com/tky823/Audyn/releases/download/v0.3.0/test_official_musicfm.pth"  # noqa: E501
    path = os.path.join(audyn_test_cache_dir, "test_official_musicfm.pth")
    download_file_from_github_release(url, path)

    data = torch.load(path, weights_only=True)

    # FMA
    spectrogram = data["fma"]["spectrogram"]
    expected_logits = data["fma"]["logits"]

    model = MusicFM.build_from_pretrained("musicfm_fma")
    model.eval()

    with torch.no_grad():
        logits = model(spectrogram)

    allclose(logits, expected_logits, atol=1e-4)

    # MSD
    spectrogram = data["msd"]["spectrogram"]
    expected_logits = data["msd"]["logits"]

    model = MusicFM.build_from_pretrained("musicfm_msd")
    model.eval()

    with torch.no_grad():
        logits = model(spectrogram)

    allclose(logits, expected_logits, atol=1e-4)


def test_musicfm_masked_token_model() -> None:
    # regression test
    url = "https://github.com/tky823/Audyn/releases/download/v0.3.0/test_official_musicfm.pth"  # noqa: E501
    path = os.path.join(audyn_test_cache_dir, "test_official_musicfm.pth")
    download_file_from_github_release(url, path)

    data = torch.load(path, weights_only=True)

    # FMA
    spectrogram = data["fma"]["spectrogram"]
    expected_logits = data["fma"]["logits"]

    model = MusicFMMaskedTokenModel.build_from_pretrained("musicfm_fma")
    projector = model.projector
    model.masker.mask_rate = 0
    model.eval()

    batch_size, _, n_frames = spectrogram.size()

    with torch.no_grad():
        logits, indices, masking_mask = model(spectrogram)

    assert masking_mask.size() == (batch_size, n_frames // projector.downsample_rate)
    assert torch.equal(data["fma"]["indices"], indices)
    allclose(logits, expected_logits, atol=1e-4)

    # MSD
    spectrogram = data["msd"]["spectrogram"]
    expected_logits = data["msd"]["logits"]

    model = MusicFMMaskedTokenModel.build_from_pretrained("musicfm_msd")
    projector = model.projector
    model.masker.mask_rate = 0
    model.eval()

    batch_size, _, n_frames = spectrogram.size()

    with torch.no_grad():
        logits, indices, masking_mask = model(spectrogram)

    assert masking_mask.size() == (batch_size, n_frames // projector.downsample_rate)
    assert torch.equal(data["msd"]["indices"], indices)
    allclose(logits, expected_logits, atol=1e-4)
