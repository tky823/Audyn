import pytest
import torch
from audyn_test.utils.github import retrieve_git_branch

role_parameters = [
    "teacher",
    "student",
]


@pytest.mark.parametrize("role", role_parameters)
def test_music_tagging_transformer(role: str) -> None:
    torch.manual_seed(0)

    branch = retrieve_git_branch()

    repo = "tky823/Audyn"
    model = "music_tagging_transformer"
    transform = "music_tagging_transformer_melspectrogram"
    batch_size = 4
    timesteps = 30 * 22050
    n_bins = 128

    if branch is not None and branch != "main":
        repo = repo + ":" + branch

    transform = torch.hub.load(
        repo,
        transform,
        skip_validation=False,
    )
    model = torch.hub.load(
        repo,
        model,
        skip_validation=False,
        role=role,
    )

    input = torch.randn((batch_size, timesteps))

    with torch.no_grad():
        spectrogram = transform(input)
        output = model(spectrogram)

    assert spectrogram.size()[:2] == (batch_size, n_bins)
    assert output.size() == (batch_size, 50)
