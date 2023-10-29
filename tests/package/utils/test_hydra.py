import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import SGD

from audyn.utils.hydra.utils import instantiate_lr_scheduler, instantiate_optimizer


def test_instantiate_optimizer() -> None:
    seed = 0
    batch_size = 4
    in_features, out_features = 3, 2
    lr = 0.1

    class DummyModel(nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()

            self.linear = nn.Linear(in_features, out_features)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            output = self.linear(input)

            return output

    torch.manual_seed(seed)

    model_by_config = DummyModel(in_features, out_features)
    config = OmegaConf.create({"_target_": "torch.optim.SGD", "lr": lr})
    optimizer = instantiate_optimizer(config, model_by_config.parameters())

    input = torch.randn((batch_size, in_features))
    target = torch.randn((batch_size, out_features))
    output = model_by_config(input)
    loss = torch.mean(output - target)
    loss.backward()
    optimizer.step()

    torch.manual_seed(seed)

    model_by_list_dict = DummyModel(in_features, out_features)
    config = OmegaConf.create(
        [
            {
                "name": "sgd",
                "optimizer": {"_target_": "torch.optim.SGD", "lr": lr},
                "modules": ["linear"],
            }
        ]
    )
    optimizer = instantiate_optimizer(config, model_by_list_dict)

    input = torch.randn((batch_size, in_features))
    target = torch.randn((batch_size, out_features))
    output = model_by_list_dict(input)
    loss = torch.mean(output - target)
    loss.backward()
    optimizer.step()

    for p_by_config, p_by_list_dict in zip(
        model_by_config.parameters(), model_by_list_dict.parameters()
    ):
        assert torch.allclose(p_by_config, p_by_list_dict)


@pytest.mark.parametrize("is_null", [True, False])
def test_instantiate_lr_scheduler(is_null: bool) -> None:
    in_features, out_features = 3, 2
    lr = 0.1

    if is_null:
        config = OmegaConf.create({})
    else:
        config = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 5})

    model = nn.Linear(in_features, out_features)
    optimizer = SGD(model.parameters(), lr=lr)
    lr_scheduler = instantiate_lr_scheduler(config, optimizer)

    if is_null:
        assert lr_scheduler is None
    else:
        assert lr_scheduler is not None
