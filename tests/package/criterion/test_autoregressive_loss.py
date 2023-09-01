import torch
import torch.nn as nn

from audyn.criterion.autoregressive import AutoRegressiveWrapper


def test_nll_loss() -> None:
    torch.manual_seed(0)

    batch_size, num_classes, length = 4, 3, 16

    input = torch.randn((batch_size, num_classes, length), dtype=torch.float)
    target = torch.randint(0, num_classes, (batch_size, length), dtype=torch.long)
    criterion = AutoRegressiveWrapper(nn.CrossEntropyLoss(reduction="mean"), dim=-1)
    loss = criterion(input, target)

    assert loss.size() == ()
