from os.path import dirname, join, realpath

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import SGD

from audyn.criterion import MultiCriteria
from audyn.modules.vqvae import VectorQuantizer
from audyn.optim.lr_scheduler import _DummyLRScheduler
from audyn.utils._hydra.utils import (
    instantiate_criterion,
    instantiate_lr_scheduler,
    instantiate_optimizer,
)

dummy_conf_dir = join(dirname(realpath(__file__)), "_conf_dummy")


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

    # optimizer for VectorQuantizer
    codebook_size = 3
    embedding_dim = 4

    vector_quantizer = VectorQuantizer(codebook_size, embedding_dim)

    config = OmegaConf.create(
        {
            "_target_": "audyn.optim.optimizer.ExponentialMovingAverageCodebookOptimizer",
        },
    )
    optimizer = instantiate_optimizer(config, vector_quantizer)


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
        assert isinstance(lr_scheduler, _DummyLRScheduler)
    else:
        assert lr_scheduler is not None


@pytest.mark.parametrize(
    "config_name",
    [
        "dummy_dict_instantiation.yaml",
        "dummy_list_instantiation.yaml",
    ],
)
def test_instantiate_criterion(config_name: str) -> None:
    config_path = join(dummy_conf_dir, "criterion", config_name)
    config = OmegaConf.load(config_path)
    criterion = instantiate_criterion(config)

    assert isinstance(criterion, MultiCriteria)
