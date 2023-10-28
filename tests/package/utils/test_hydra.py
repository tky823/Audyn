import pytest
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import SGD

from audyn.utils.hydra.utils import instantiate_lr_scheduler


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
