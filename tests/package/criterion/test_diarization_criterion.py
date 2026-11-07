import pytest
import torch
import torch.nn as nn
from audyn_test import allclose

from audyn.criterion.diarization import PowersetPITLoss


def test_powerset_pit_loss() -> None:
    batch_size = 2
    num_speakers = 3

    # Targets include overlapping speakers and a frame beyond max_subset.
    target = torch.zeros((batch_size, num_speakers, 5))
    target[0, 0, :2] = 1
    target[0, 1, 1:] = 1
    target[0, 2, 3:4] = 1
    target[1, 0, 2:4] = 1
    target[1, 1, 0:3] = 1
    target[1, 2, 2:3] = 1

    logits = torch.full((2, 7, 5), -5.0)
    predicted_classes = torch.tensor([[1, 2, 3, 1, 2], [1, 2, 3, 1, 2]])
    logits.scatter_(dim=1, index=predicted_classes.unsqueeze(dim=1), value=5.0)
    logits.requires_grad_()

    criterion = PowersetPITLoss(num_classes=num_speakers, max_subset=2)
    loss = criterion(logits, target)
    # Batch 0 has a tie in the surrogate MSE loss. Permutations are enumerated
    # with the identity first, and argmin selects that first minimum. Batch 1
    # also selects the identity, so these class IDs use the original target order.
    expected_classes = torch.tensor([[1, 4, 2, 6, 2], [2, 2, 4, 1, 0]])
    expected_loss = nn.CrossEntropyLoss()(logits, expected_classes)

    assert loss.size() == ()
    allclose(loss, expected_loss)


def test_powerset_pit_loss_permutation_criteria() -> None:
    target = torch.tensor([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    logits = torch.tensor(
        [
            [[-20.0, -20.0], [0.1, 10.0], [0.0, 0.0]],
            [[-20.0, -20.0], [10.0, 0.1], [0.0, 0.0]],
        ],
        requires_grad=True,
    )

    identity = torch.tensor([[1, 2], [1, 2]])
    exact = torch.tensor([[2, 1], [1, 2]])

    default_loss = PowersetPITLoss(2, 1)(logits, target)
    mse_loss = PowersetPITLoss(2, 1, approximate="mse")(logits, target)
    mae_loss = PowersetPITLoss(2, 1, approximate="mae")(logits, target)
    exact_loss = PowersetPITLoss(2, 1, approximate="none")(logits, target)
    true_loss = PowersetPITLoss(2, 1, approximate=True)(logits, target)
    false_loss = PowersetPITLoss(2, 1, approximate=False)(logits, target)

    allclose(default_loss, nn.CrossEntropyLoss()(logits, identity))
    allclose(mse_loss, default_loss)
    allclose(mae_loss, default_loss)
    allclose(true_loss, mse_loss)
    allclose(exact_loss, nn.CrossEntropyLoss()(logits, exact))
    allclose(false_loss, exact_loss)
    assert exact_loss < default_loss

    exact_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("method", ["unknown", "exact"])
def test_powerset_pit_loss_invalid_approximation(method: str) -> None:
    with pytest.raises(ValueError, match="approximate"):
        PowersetPITLoss(2, 1, approximate=method)


def test_powerset_pit_loss_invalid_callable_approximation() -> None:
    with pytest.raises(ValueError, match="approximate"):
        PowersetPITLoss(2, 1, approximate=lambda prediction, reference: prediction.new_zeros(1))
