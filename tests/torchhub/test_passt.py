import torch
from audyn_test.utils.github import retrieve_git_branch


def test_passt() -> None:
    torch.manual_seed(0)

    branch = retrieve_git_branch()

    repo = "tky823/Audyn"
    model = "passt_base"
    batch_size = 4
    n_bins, n_frames = 128, 998

    if branch is not None and branch != "main":
        repo = repo + ":" + branch

    model = torch.hub.load(
        repo,
        model,
        skip_validation=False,
        n_frames=n_frames,
    )

    input = torch.randn((batch_size, n_bins, n_frames))

    with torch.no_grad():
        output = model(input)

    assert output.size() == (batch_size, 527)
