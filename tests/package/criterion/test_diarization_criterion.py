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
    # Batch 0 has a tie in the surrogate BCE loss. Permutations are enumerated
    # with the identity first, and argmin selects that first minimum. Batch 1
    # also selects the identity, so these class IDs use the original target order.
    expected_classes = torch.tensor([[1, 4, 2, 6, 2], [2, 2, 4, 1, 0]])
    expected_loss = nn.CrossEntropyLoss()(logits, expected_classes)

    assert loss.size() == ()
    allclose(loss, expected_loss)
