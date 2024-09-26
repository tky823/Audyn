import pytest
import torch
from dummy.utils.github import retrieve_git_branch

models = [
    "teacher_music_tagging_transformer",
    "student_music_tagging_transformer",
]


@pytest.mark.parametrize("model", models)
def test_music_tagging_transformer(model: str) -> None:
    torch.manual_seed(0)

    branch = retrieve_git_branch()

    repo = "tky823/Audyn"
    batch_size = 4
    timesteps = 30 * 22050

    if branch is not None and branch != "main":
        repo = repo + ":" + branch

    model = torch.hub.load(
        repo,
        model,
        skip_validation=False,
    )

    input = torch.randn((batch_size, timesteps))

    with torch.no_grad():
        output = model(input)

    assert output.size() == (batch_size, 10)
