import itertools
import math

import pytest
import torch

from audyn.functional.assignment import optimal_assignment

IS_MPS_AVAILABLE = torch.backends.mps.is_available()


def test_optimal_assignment() -> None:
    generator = torch.Generator()
    generator.manual_seed(0)
    cost = torch.randn(8, 4, 4, generator=generator)

    assignment = optimal_assignment(cost)

    assert assignment.shape == (8, 4)
    assert assignment.dtype == torch.long

    rows = torch.arange(4)

    for batch_idx in range(cost.size(0)):
        columns = assignment[batch_idx]

        assert sorted(columns.tolist()) == list(range(4))

        output = cost[batch_idx, rows, columns].sum().item()
        expected_output = float("inf")

        for permutation in itertools.permutations(range(4)):
            _cost = cost[batch_idx, rows, permutation]
            _output = _cost.sum().item()
            expected_output = min(expected_output, _output)

        assert math.isclose(output, expected_output, abs_tol=1e-5)


def test_unbatched_noncontiguous_assignment() -> None:
    cost = torch.tensor([[3.0, 1.0], [2.0, 4.0]])
    cost = cost.transpose(0, 1)

    assert not cost.is_contiguous()

    assignment = optimal_assignment(cost)

    assert assignment.shape == (2,)
    assert assignment.dtype == torch.long
    assert assignment.tolist() == [1, 0]


def test_empty_assignment() -> None:
    assert optimal_assignment(torch.empty(0, 0)).shape == (0,)
    assert optimal_assignment(torch.empty(0, 3, 3)).shape == (0, 3)


def test_invalid_input() -> None:
    with pytest.raises(ValueError, match="2- or 3-dimensional"):
        optimal_assignment(torch.empty(3))

    with pytest.raises(ValueError, match="square"):
        optimal_assignment(torch.empty(2, 3))

    with pytest.raises(ValueError, match="dtype"):
        optimal_assignment(torch.ones(2, 2, dtype=torch.float64))


@pytest.mark.skipif(not IS_MPS_AVAILABLE, reason="MPS is unavailable")
def test_result_device() -> None:
    cost = torch.tensor([[[3.0, 1.0], [2.0, 4.0]]], device="mps")
    assignment = optimal_assignment(cost)

    assert assignment.device == cost.device
    assert assignment.tolist() == [[1, 0]]
    assert assignment.tolist() == [[1, 0]]
