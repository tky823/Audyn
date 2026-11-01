import pytest
import torch

from audyn.functional.diarization import multilabel_to_powerset, powerset_to_multilabel


def test_multilabel_to_powerset() -> None:
    # {}, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}
    activity = torch.tensor(
        [
            [0, 1, 0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 1],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    expected = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 4]])

    actual = multilabel_to_powerset(activity, num_classes=3, max_subset=2)
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.long

    lookup = powerset_to_multilabel(
        torch.eye(7).unsqueeze(-1), num_classes=3, max_subset=2
    ).squeeze(-1)
    assert lookup.dtype == torch.long
    assert torch.equal(multilabel_to_powerset(activity, lookup), expected)
    assert torch.equal(
        multilabel_to_powerset(activity, powerset_to_multilabel=lookup), expected
    )
    assert torch.equal(multilabel_to_powerset(activity, 3, max_subset=2), expected)

    decoded = powerset_to_multilabel(
        torch.nn.functional.one_hot(actual[:, :-1], num_classes=7).transpose(1, 2), lookup
    )
    assert torch.equal(decoded, activity[:, :, :-1])
    assert decoded.dtype == torch.long
    assert powerset_to_multilabel(
        torch.nn.functional.one_hot(actual[:, :-1], num_classes=7).transpose(1, 2),
        lookup.float(),
    ).dtype == torch.long


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((), {}),
        ((0, 0), {}),
        ((3, -1), {}),
        ((3, 4), {}),
        ((torch.zeros(7),), {}),
        ((torch.zeros(7, 2),), {}),
    ],
)
def test_multilabel_to_powerset_invalid_arguments(args, kwargs) -> None:
    with pytest.raises(ValueError):
        multilabel_to_powerset(torch.zeros(1, 3, 2), *args, **kwargs)


def test_multilabel_to_powerset_invalid_shape() -> None:
    with pytest.raises(ValueError):
        multilabel_to_powerset(torch.zeros(3, 2), 3, 2)
