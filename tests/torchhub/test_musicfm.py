import pytest
import torch
from audyn_test.utils.github import retrieve_git_branch

dataset_parameters = [
    "fma",
    "msd",
]


@pytest.mark.parametrize("dataset", dataset_parameters)
def test_music_tagging_transformer(dataset: str) -> None:
    torch.manual_seed(0)

    branch = retrieve_git_branch()

    repo = "tky823/Audyn"
    model = "musicfm"
    transform = "musicfm_melspectrogram"
    batch_size = 4
    timesteps = 10 * 24000
    n_bins = 128

    if branch is not None and branch != "main":
        repo = repo + ":" + branch

    transform = torch.hub.load(
        repo,
        transform,
        skip_validation=False,
        dataset=dataset,
    )
    model = torch.hub.load(
        repo,
        model,
        skip_validation=False,
        dataset=dataset,
    )

    input = torch.randn((batch_size, timesteps))

    with torch.no_grad():
        spectrogram = transform(input)
        output = model(spectrogram)

    assert spectrogram.size()[:2] == (batch_size, n_bins)
    assert output.size() == (batch_size, timesteps // 960, 4096)
