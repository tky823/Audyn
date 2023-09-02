import os
import shutil
import sys
import uuid
from os.path import dirname, join, realpath, relpath
from typing import List

import hydra
import pytest
import torch
from pytest import MonkeyPatch

import audyn
from audyn.utils.data import BaseDataLoaders
from audyn.utils.driver import BaseTrainer

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))


config_template_path = join(dirname(realpath(audyn.__file__)), "utils", "driver", "_conf_template")
config_name = "config"


@pytest.mark.parametrize("use_ema", [True, False])
def test_base_trainer(monkeypatch: MonkeyPatch, use_ema: bool) -> None:
    DATA_SIZE = 20
    BATCH_SIZE = 2
    INITIAL_ITERATION = 3

    temp_dir = str(uuid.uuid4())
    os.makedirs(temp_dir, exist_ok=False)
    monkeypatch.chdir(temp_dir)

    overrides_conf_dir = relpath(join(dirname(realpath(__file__)), "_conf_dummy"), os.getcwd())
    exp_dir = "./exp"

    if use_ema:
        optimizer_name = "dummy"
    else:
        optimizer_name = "dummy_ema"

    with hydra.initialize(
        version_base="1.2",
        config_path=relpath(config_template_path, dirname(realpath(__file__))),
        job_name="test_driver",
    ):
        config = hydra.compose(
            config_name="config",
            overrides=create_dummy_override(
                overrides_conf_dir=overrides_conf_dir,
                exp_dir=exp_dir,
                data_size=DATA_SIZE,
                batch_size=BATCH_SIZE,
                iterations=INITIAL_ITERATION,
                optimizer=optimizer_name,
            ),
            return_hydra_config=True,
        )

    train_dataset = hydra.utils.instantiate(config.train.dataset.train)
    validation_dataset = hydra.utils.instantiate(config.train.dataset.validation)

    train_loader = hydra.utils.instantiate(
        config.train.dataloader.train,
        train_dataset,
    )
    validation_loader = hydra.utils.instantiate(
        config.train.dataloader.validation,
        validation_dataset,
    )
    loaders = BaseDataLoaders(train_loader, validation_loader)

    model = hydra.utils.instantiate(config.model)
    optimizer = hydra.utils.instantiate(config.optimizer, model.parameters())
    lr_scheduler = hydra.utils.instantiate(config.lr_scheduler, optimizer)
    criterion = hydra.utils.instantiate(config.criterion)

    trainer = BaseTrainer(
        loaders,
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        config=config,
    )
    trainer.run()

    target_grad = torch.tensor(
        list(range(BATCH_SIZE * INITIAL_ITERATION - BATCH_SIZE, BATCH_SIZE * INITIAL_ITERATION)),
        dtype=torch.float,
    )
    target_grad = -torch.mean(target_grad)

    assert torch.allclose(model.linear.weight.grad.data, target_grad)

    with hydra.initialize(
        version_base="1.2",
        config_path=relpath(config_template_path, dirname(realpath(__file__))),
        job_name="test_driver",
    ):
        config = hydra.compose(
            config_name="config",
            overrides=create_dummy_override(
                overrides_conf_dir=overrides_conf_dir,
                exp_dir=exp_dir,
                data_size=DATA_SIZE,
                batch_size=BATCH_SIZE,
                iterations=len(train_loader),
                optimizer=optimizer_name,
                continue_from=f"{exp_dir}/model/iteration{INITIAL_ITERATION}.pth",
            ),
            return_hydra_config=True,
        )

    trainer = BaseTrainer(
        loaders,
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        config=config,
    )
    trainer.run()

    target_grad = torch.tensor(list(range(DATA_SIZE - BATCH_SIZE, DATA_SIZE)), dtype=torch.float)
    target_grad = -torch.mean(target_grad)

    assert torch.allclose(model.linear.weight.grad.data, target_grad)

    monkeypatch.undo()

    shutil.rmtree(temp_dir)


def create_dummy_override(
    overrides_conf_dir: str,
    exp_dir: str,
    data_size: int,
    batch_size: int = 1,
    iterations: int = 1,
    optimizer: str = "dummy",
    continue_from: str = "",
) -> List[str]:
    return [
        "train=dummy",
        "model=dummy",
        f"optimizer={optimizer}",
        "lr_scheduler=dummy",
        "criterion=dummy",
        f"hydra.searchpath=[{overrides_conf_dir}]",
        "hydra.job.num=1",
        f"hydra.runtime.output_dir={exp_dir}/log",
        f"train.dataset.train.size={data_size}",
        f"train.dataset.validation.size={data_size}",
        f"train.dataloader.train.batch_size={batch_size}",
        f"train.dataloader.validation.batch_size={batch_size}",
        f"train.output.exp_dir={exp_dir}",
        f"train.resume.continue_from={continue_from}",
        f"train.steps.iterations={iterations}",
    ]
