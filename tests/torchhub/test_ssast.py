import pytest
import torch
from dummy.utils.github import retrieve_git_branch


@pytest.mark.parametrize("token_unit", ["patch", "frame"])
def test_multitask_ssast_mpm(token_unit: str) -> None:
    torch.manual_seed(0)

    branch = retrieve_git_branch()

    repo = "tky823/Audyn"
    model = "multitask_ssast_base_400"
    batch_size = 4
    n_bins, n_frames = 128, 1024

    if branch is not None and branch != "main":
        repo = repo + ":" + branch

    model = torch.hub.load(
        repo,
        model,
        skip_validation=False,
        token_unit=token_unit,
    )

    input = torch.randn((batch_size, n_bins, n_frames))

    with torch.no_grad():
        output = model(input)

    reconstruction_output, classification_output = output
    reconstruction_output, reconstruction_target, reconstruction_length = reconstruction_output
    classification_output, classification_target, classification_length = classification_output

    assert reconstruction_output.size() == (batch_size, 400, 256)
    assert reconstruction_target.size() == (batch_size, 400, 256)
    assert reconstruction_length.size() == (batch_size,)

    assert classification_output.size() == (batch_size, 400, 256)
    assert classification_target.size() == (batch_size, 400, 256)
    assert classification_length.size() == (batch_size,)


@pytest.mark.parametrize("token_unit", ["patch", "frame"])
def test_ssast(token_unit: str) -> None:
    torch.manual_seed(0)

    branch = retrieve_git_branch()

    repo = "tky823/Audyn"
    model = "ssast_base_400"
    batch_size = 4
    n_bins, n_frames = 128, 100

    if branch is not None and branch != "main":
        repo = repo + ":" + branch

    if token_unit == "patch":
        stride = (10, 10)
    elif token_unit == "frame":
        stride = (n_bins, 1)
    else:
        raise ValueError(f"{token_unit} is not supported as token_unit.")

    model = torch.hub.load(
        repo,
        model,
        skip_validation=False,
        token_unit=token_unit,
        stride=stride,
        n_frames=n_frames,
    )

    input = torch.randn((batch_size, n_bins, n_frames))

    with torch.no_grad():
        output = model(input)

    if token_unit == "patch":
        assert output.size() == (batch_size, 12 * 9 + 2, 768)
    elif token_unit == "frame":
        assert output.size() == (batch_size, 1 * 99 + 2, 768)
    else:
        raise ValueError(f"{token_unit} is not supported as token_unit.")
