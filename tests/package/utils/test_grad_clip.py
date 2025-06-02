import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

from audyn.utils.clip_grad import GradClipper


@pytest.mark.parametrize("mode", ["value", "norm"])
def test_grad_clipper(mode: str) -> None:
    torch.manual_seed(0)

    batch_size = 4
    in_features, out_features = 3, 2
    clip_value = 0.1
    max_norm = 0.1

    model = nn.Linear(in_features, out_features)
    optimizer = Adam(model.parameters())

    if mode == "value":
        kwargs = {"clip_value": clip_value}
    elif mode == "norm":
        kwargs = {"max_norm": max_norm}
    else:
        raise ValueError("Invalid mode is given.")

    grad_clipper = GradClipper(model.parameters(), mode=mode, **kwargs)

    input = torch.randn((batch_size, in_features))
    target = torch.randn((batch_size, out_features))

    output = model(input)
    loss = torch.mean((output - target) ** 2)

    optimizer.zero_grad()
    loss.backward()
    grad_clipper.step()

    for p in model.parameters():
        if mode == "value":
            assert torch.all(torch.abs(p.grad) <= clip_value)
        elif mode == "norm":
            norm = torch.linalg.vector_norm(p.grad.view(-1))
            assert torch.all(norm <= max_norm)
        else:
            raise ValueError("Invalid mode is given.")

    optimizer.step()
