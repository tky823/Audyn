import os
import sys
import tempfile
from os.path import dirname, join, realpath, relpath
from typing import Any, Dict, List

import hydra
import pytest
import torch
from pytest import MonkeyPatch

import audyn
from audyn.criterion.gan import GANCriterion
from audyn.models.gan import BaseGAN
from audyn.optim.lr_scheduler import GANLRScheduler
from audyn.optim.optimizer import GANOptimizer
from audyn.utils import instantiate_model
from audyn.utils.data import BaseDataLoaders, default_collate_fn, make_noise
from audyn.utils.driver import BaseGenerator, BaseTrainer, FeatToWaveTrainer, GANTrainer

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))


config_template_path = join(dirname(realpath(audyn.__file__)), "utils", "driver", "_conf_template")
config_name = "config"


@pytest.mark.parametrize("use_ema", [True, False])
def test_base_drivers(monkeypatch: MonkeyPatch, use_ema: bool) -> None:
    """Test BaseTrainer and BaseGenerator."""
    DATA_SIZE = 20
    BATCH_SIZE = 2
    INITIAL_ITERATION = 3

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        monkeypatch.chdir(temp_dir)

        overrides_conf_dir = relpath(join(dirname(realpath(__file__)), "_conf_dummy"), os.getcwd())
        exp_dir = "./exp"

        train_name = "dummy"
        model_name = "dummy"
        criterion_name = "dummy"

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
                    train=train_name,
                    model=model_name,
                    criterion=criterion_name,
                    optimizer=optimizer_name,
                ),
                return_hydra_config=True,
            )

        train_dataset = hydra.utils.instantiate(config.train.dataset.train)
        validation_dataset = hydra.utils.instantiate(config.train.dataset.validation)
        test_dataset = hydra.utils.instantiate(config.test.dataset.test)

        train_loader = hydra.utils.instantiate(
            config.train.dataloader.train,
            train_dataset,
        )
        validation_loader = hydra.utils.instantiate(
            config.train.dataloader.validation,
            validation_dataset,
        )
        test_loader = hydra.utils.instantiate(
            config.test.dataloader.test,
            test_dataset,
        )
        loaders = BaseDataLoaders(train_loader, validation_loader)

        model = instantiate_model(config.model)
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
        trainer.writer.flush()

        target_grad = torch.tensor(
            list(
                range(BATCH_SIZE * INITIAL_ITERATION - BATCH_SIZE, BATCH_SIZE * INITIAL_ITERATION)
            ),
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
                    train=train_name,
                    model=model_name,
                    criterion=criterion_name,
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
        trainer.writer.flush()

        target_grad = torch.tensor(
            list(range(DATA_SIZE - BATCH_SIZE, DATA_SIZE)), dtype=torch.float
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
                    checkpoint=f"{exp_dir}/model/last.pth",
                ),
                return_hydra_config=True,
            )

        model = instantiate_model(config.test.checkpoint)

        generator = BaseGenerator(
            test_loader,
            model,
            config=config,
        )
        generator.run()

        monkeypatch.undo()


@pytest.mark.parametrize("use_ema", [True, False])
def test_feat_to_wave_trainer(monkeypatch: MonkeyPatch, use_ema: bool):
    """Test FeatToWaveTrainer."""
    DATA_SIZE = 20
    BATCH_SIZE = 2
    INITIAL_ITERATION = 3

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        monkeypatch.chdir(temp_dir)

        overrides_conf_dir = relpath(join(dirname(realpath(__file__)), "_conf_dummy"), os.getcwd())
        exp_dir = "./exp"

        train_name = "dummy_autoregressive_feat_to_wave"
        model_name = "dummy_autoregressive"
        criterion_name = "dummy_autoregressive"

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
                    train=train_name,
                    model=model_name,
                    criterion=criterion_name,
                    optimizer=optimizer_name,
                ),
                return_hydra_config=True,
            )

        train_dataset = hydra.utils.instantiate(config.train.dataset.train)
        validation_dataset = hydra.utils.instantiate(config.train.dataset.validation)

        train_loader = hydra.utils.instantiate(
            config.train.dataloader.train,
            train_dataset,
            collate_fn=pad_collate_fn,
        )
        validation_loader = hydra.utils.instantiate(
            config.train.dataloader.validation,
            validation_dataset,
            collate_fn=pad_collate_fn,
        )
        loaders = BaseDataLoaders(train_loader, validation_loader)

        model = instantiate_model(config.model)
        optimizer = hydra.utils.instantiate(config.optimizer, model.parameters())
        lr_scheduler = hydra.utils.instantiate(config.lr_scheduler, optimizer)
        criterion = hydra.utils.instantiate(config.criterion)

        trainer = FeatToWaveTrainer(
            loaders,
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            config=config,
        )
        trainer.run()
        trainer.writer.flush()

        monkeypatch.undo()


@pytest.mark.parametrize("use_ema_generator", [True, False])
@pytest.mark.parametrize("use_ema_discriminator", [True, False])
def test_gan_trainer(
    monkeypatch: MonkeyPatch,
    use_ema_generator: bool,
    use_ema_discriminator: bool,
):
    DATA_SIZE = 20
    BATCH_SIZE = 2
    INITIAL_ITERATION = 3

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        monkeypatch.chdir(temp_dir)

        overrides_conf_dir = relpath(join(dirname(realpath(__file__)), "_conf_dummy"), os.getcwd())
        exp_dir = "./exp"

        train_name = "dummy_gan"
        model_name = "dummy_gan"
        criterion_name = "dummy_gan"

        with hydra.initialize(
            version_base="1.2",
            config_path=relpath(config_template_path, dirname(realpath(__file__))),
            job_name="test_driver",
        ):
            config = hydra.compose(
                config_name="config",
                overrides=create_dummy_gan_override(
                    overrides_conf_dir=overrides_conf_dir,
                    exp_dir=exp_dir,
                    data_size=DATA_SIZE,
                    batch_size=BATCH_SIZE,
                    iterations=INITIAL_ITERATION,
                    train=train_name,
                    model=model_name,
                    criterion=criterion_name,
                    use_ema_generator=use_ema_generator,
                    use_ema_discriminator=use_ema_discriminator,
                ),
                return_hydra_config=True,
            )

        train_dataset = hydra.utils.instantiate(
            config.train.dataset.train,
        )
        validation_dataset = hydra.utils.instantiate(
            config.train.dataset.validation,
        )

        train_loader = hydra.utils.instantiate(
            config.train.dataloader.train,
            train_dataset,
            collate_fn=gan_collate_fn,
        )
        validation_loader = hydra.utils.instantiate(
            config.train.dataloader.validation,
            validation_dataset,
            collate_fn=gan_collate_fn,
        )
        loaders = BaseDataLoaders(train_loader, validation_loader)

        generator = instantiate_model(config.model.generator)
        discriminator = instantiate_model(config.model.discriminator)

        generator_optimizer = hydra.utils.instantiate(
            config.optimizer.generator, generator.parameters()
        )
        discriminator_optimizer = hydra.utils.instantiate(
            config.optimizer.discriminator, discriminator.parameters()
        )
        generator_lr_scheduler = hydra.utils.instantiate(
            config.lr_scheduler.generator, generator_optimizer
        )
        discriminator_lr_scheduler = hydra.utils.instantiate(
            config.lr_scheduler.discriminator, discriminator_optimizer
        )
        generator_criterion = hydra.utils.instantiate(config.criterion.generator)
        discriminator_criterion = hydra.utils.instantiate(config.criterion.discriminator)

        model = BaseGAN(generator, discriminator)
        optimizer = GANOptimizer(generator_optimizer, discriminator_optimizer)
        lr_scheduler = GANLRScheduler(generator_lr_scheduler, discriminator_lr_scheduler)
        criterion = GANCriterion(generator_criterion, discriminator_criterion)

        trainer = GANTrainer(
            loaders,
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            config=config,
        )
        trainer.run()
        trainer.writer.flush()

        monkeypatch.undo()


def create_dummy_override(
    overrides_conf_dir: str,
    exp_dir: str,
    data_size: int,
    batch_size: int = 1,
    iterations: int = 1,
    train: str = "dummy",
    model: str = "dummy",
    criterion: str = "dummy",
    optimizer: str = "dummy",
    continue_from: str = "",
    checkpoint: str = "",
) -> List[str]:
    sample_rate = 16000

    return [
        f"train={train}",
        "test=dummy",
        f"model={model}",
        f"optimizer={optimizer}",
        "lr_scheduler=dummy",
        f"criterion={criterion}",
        f"hydra.searchpath=[{overrides_conf_dir}]",
        "hydra.job.num=1",
        f"hydra.runtime.output_dir={exp_dir}/log",
        f"data.audio.sample_rate={sample_rate}",
        f"train.dataset.train.size={data_size}",
        f"train.dataset.validation.size={data_size}",
        f"train.dataloader.train.batch_size={batch_size}",
        f"train.dataloader.validation.batch_size={batch_size}",
        f"train.output.exp_dir={exp_dir}",
        f"train.resume.continue_from={continue_from}",
        f"train.steps.iterations={iterations}",
        f"test.dataset.test.size={data_size}",
        f"test.checkpoint={checkpoint}",
    ]


def create_dummy_gan_override(
    overrides_conf_dir: str,
    exp_dir: str,
    data_size: int,
    batch_size: int = 1,
    iterations: int = 1,
    train: str = "dummy_gan",
    model: str = "dummy_gan",
    criterion: str = "dummy_gan",
    use_ema_generator: bool = False,
    use_ema_discriminator: bool = False,
    continue_from: str = "",
) -> List[str]:
    sample_rate = 16000
    ema_target = "audyn.optim.optimizer.ExponentialMovingAverageWrapper.build_from_optim_class"

    if use_ema_generator:
        generator_optimizer_config = [
            f"+optimizer.generator._target_={ema_target}",
            "+optimizer.generator.optimizer_class={_target_:torch.optim.Adam,_partial_:true}",
        ]
    else:
        generator_optimizer_config = [
            "+optimizer.generator._target_=torch.optim.Adam",
        ]

    if use_ema_discriminator:
        generator_discriminator_config = [
            f"+optimizer.discriminator._target_={ema_target}",
            "+optimizer.discriminator.optimizer_class={_target_:torch.optim.Adam,_partial_:true}",
        ]
    else:
        generator_discriminator_config = [
            "+optimizer.discriminator._target_=torch.optim.Adam",
        ]

    overridden_config = [
        f"train={train}",
        "test=dummy",
        f"model={model}",
        "optimizer=gan",
        "lr_scheduler=dummy_gan",
        f"criterion={criterion}",
        f"hydra.searchpath=[{overrides_conf_dir}]",
        "hydra.job.num=1",
        f"hydra.runtime.output_dir={exp_dir}/log",
        f"data.audio.sample_rate={sample_rate}",
        f"train.dataset.train.size={data_size}",
        f"train.dataset.validation.size={data_size}",
        f"train.dataloader.train.batch_size={batch_size}",
        f"train.dataloader.validation.batch_size={batch_size}",
        f"train.output.exp_dir={exp_dir}",
        f"train.resume.continue_from={continue_from}",
        f"train.steps.iterations={iterations}",
    ]
    overridden_config = (
        overridden_config + generator_optimizer_config + generator_discriminator_config
    )

    return overridden_config


def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    dict_batch = default_collate_fn(batch)
    dict_batch["initial_state"] = torch.zeros((dict_batch["input"].size(0), 1), dtype=torch.float)

    if "target" in dict_batch:
        max_length = dict_batch["target"].size(-1)
    else:
        max_length = dict_batch["input"].size(-1)

    dict_batch["max_length"] = max_length

    return dict_batch


def gan_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    dict_batch = pad_collate_fn(batch)
    dict_batch = make_noise(dict_batch, key_mapping={"input": "noise"})

    return dict_batch
