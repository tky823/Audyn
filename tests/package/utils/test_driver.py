import os
import tempfile
from os.path import dirname, join, realpath, relpath
from typing import Any, Dict, List

import hydra
import pytest
import torch
from dummy import allclose
from pytest import MonkeyPatch

import audyn
from audyn.criterion.gan import GANCriterion
from audyn.models.gan import BaseGAN
from audyn.optim.lr_scheduler import GANLRScheduler
from audyn.optim.optimizer import GANOptimizer
from audyn.utils import (
    instantiate_cascade_text_to_wave,
    instantiate_lr_scheduler,
    instantiate_model,
    instantiate_optimizer,
    setup_system,
)
from audyn.utils.data import BaseDataLoaders, default_collate_fn, make_noise
from audyn.utils.driver import (
    BaseGenerator,
    BaseTrainer,
    CascadeTextToWaveGenerator,
    FeatToWaveTrainer,
    GANTrainer,
    TextToFeatTrainer,
)
from audyn.utils.model import set_device

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

        setup_system(config)

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
        model = set_device(
            model,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )
        optimizer = instantiate_optimizer(config.optimizer, model.parameters())
        lr_scheduler = instantiate_lr_scheduler(config.lr_scheduler, optimizer)
        criterion = hydra.utils.instantiate(config.criterion)
        criterion = set_device(
            criterion,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
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
            list(
                range(BATCH_SIZE * INITIAL_ITERATION - BATCH_SIZE, BATCH_SIZE * INITIAL_ITERATION)
            ),
            dtype=torch.float,
        )
        target_grad = -torch.mean(target_grad)

        allclose(model.linear.weight.grad.data, target_grad)

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

        allclose(model.linear.weight.grad.data, target_grad)

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
        model = set_device(
            model,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )

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

        setup_system(config)

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
        model = set_device(
            model,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )
        optimizer = instantiate_optimizer(config.optimizer, model.parameters())
        lr_scheduler = instantiate_lr_scheduler(config.lr_scheduler, optimizer)
        criterion = hydra.utils.instantiate(config.criterion)
        criterion = set_device(
            criterion,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )

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

        setup_system(config)

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
        generator = set_device(
            generator,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )
        discriminator = set_device(
            discriminator,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )

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
        generator_criterion = set_device(
            generator_criterion,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )
        discriminator_criterion = set_device(
            discriminator_criterion,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )

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


def test_cascade_text_to_wave(monkeypatch: MonkeyPatch) -> None:
    DATA_SIZE = 20
    BATCH_SIZE = 2
    ITERATIONS = 3
    VOCAB_SIZE = 10
    N_MELS = 5
    TEXT_TO_FEAT_UP_SCALE = 2
    FEAT_TO_WAVE_UP_SCALE = 2

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        monkeypatch.chdir(temp_dir)

        overrides_conf_dir = relpath(join(dirname(realpath(__file__)), "_conf_dummy"), os.getcwd())
        text_to_feat_exp_dir = "./exp/text-to-feat"
        feat_to_wave_exp_dir = "./exp/feat-to-wave"
        text_to_wave_exp_dir = "./exp/text-to-wave"

        # text-to-feat
        with hydra.initialize(
            version_base="1.2",
            config_path=relpath(config_template_path, dirname(realpath(__file__))),
            job_name="test_driver",
        ):
            config = hydra.compose(
                config_name="config",
                overrides=create_dummy_text_to_feat_override(
                    overrides_conf_dir=overrides_conf_dir,
                    exp_dir=text_to_feat_exp_dir,
                    data_size=DATA_SIZE,
                    batch_size=BATCH_SIZE,
                    iterations=ITERATIONS,
                    vocab_size=VOCAB_SIZE,
                    n_mels=N_MELS,
                    up_scale=TEXT_TO_FEAT_UP_SCALE,
                ),
                return_hydra_config=True,
            )

        setup_system(config)

        train_dataset = hydra.utils.instantiate(config.train.dataset.train)
        validation_dataset = hydra.utils.instantiate(config.train.dataset.validation)

        train_loader = hydra.utils.instantiate(
            config.train.dataloader.train,
            train_dataset,
            collate_fn=default_collate_fn,
        )
        validation_loader = hydra.utils.instantiate(
            config.train.dataloader.validation,
            validation_dataset,
            collate_fn=default_collate_fn,
        )
        loaders = BaseDataLoaders(train_loader, validation_loader)

        model = instantiate_model(config.model)
        model = set_device(
            model,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )
        optimizer = instantiate_optimizer(config.optimizer, model.parameters())
        lr_scheduler = instantiate_lr_scheduler(config.lr_scheduler, optimizer)
        criterion = hydra.utils.instantiate(config.criterion)
        criterion = set_device(
            criterion,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )

        trainer = TextToFeatTrainer(
            loaders,
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            config=config,
        )
        trainer.run()
        trainer.writer.flush()

        # feat-to-wave
        with hydra.initialize(
            version_base="1.2",
            config_path=relpath(config_template_path, dirname(realpath(__file__))),
            job_name="test_driver",
        ):
            config = hydra.compose(
                config_name="config",
                overrides=create_dummy_feat_to_wave_override(
                    overrides_conf_dir=overrides_conf_dir,
                    exp_dir=feat_to_wave_exp_dir,
                    data_size=DATA_SIZE,
                    batch_size=BATCH_SIZE,
                    iterations=ITERATIONS,
                    n_mels=N_MELS,
                    up_scale=FEAT_TO_WAVE_UP_SCALE,
                ),
                return_hydra_config=True,
            )

        setup_system(config)

        train_dataset = hydra.utils.instantiate(config.train.dataset.train)
        validation_dataset = hydra.utils.instantiate(config.train.dataset.validation)

        train_loader = hydra.utils.instantiate(
            config.train.dataloader.train,
            train_dataset,
            collate_fn=default_collate_fn,
        )
        validation_loader = hydra.utils.instantiate(
            config.train.dataloader.validation,
            validation_dataset,
            collate_fn=default_collate_fn,
        )
        loaders = BaseDataLoaders(train_loader, validation_loader)

        model = instantiate_model(config.model)
        model = set_device(
            model,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )
        optimizer = instantiate_optimizer(config.optimizer, model.parameters())
        lr_scheduler = instantiate_lr_scheduler(config.lr_scheduler, optimizer)
        criterion = hydra.utils.instantiate(config.criterion)
        criterion = set_device(
            criterion,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )

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

        # text-to-wave
        text_to_feat_checkpoint = os.path.join(text_to_feat_exp_dir, "model", "last.pth")
        feat_to_wave_checkpoint = os.path.join(feat_to_wave_exp_dir, "model", "last.pth")

        # text-to-wave
        with hydra.initialize(
            version_base="1.2",
            config_path=relpath(config_template_path, dirname(realpath(__file__))),
            job_name="test_driver",
        ):
            config = hydra.compose(
                config_name="config",
                overrides=create_dummy_text_to_wave_override(
                    overrides_conf_dir=overrides_conf_dir,
                    exp_dir=text_to_wave_exp_dir,
                    data_size=DATA_SIZE,
                    batch_size=BATCH_SIZE,
                    vocab_size=VOCAB_SIZE,
                    up_scale=TEXT_TO_FEAT_UP_SCALE * FEAT_TO_WAVE_UP_SCALE,
                    text_to_feat_checkpoint=text_to_feat_checkpoint,
                    feat_to_wave_checkpoint=feat_to_wave_checkpoint,
                ),
                return_hydra_config=True,
            )

        test_dataset = hydra.utils.instantiate(config.test.dataset.test)

        test_loader = hydra.utils.instantiate(
            config.test.dataloader.test,
            test_dataset,
            collate_fn=default_collate_fn,
        )

        model = instantiate_cascade_text_to_wave(
            config.model,
            text_to_feat_checkpoint=config.test.checkpoint.text_to_feat,
            feat_to_wave_checkpoint=config.test.checkpoint.feat_to_wave,
        )
        model = set_device(
            model,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )

        generator = CascadeTextToWaveGenerator(
            test_loader,
            model,
            config=config,
        )
        generator.run()

        monkeypatch.undo()


@pytest.mark.parametrize(
    "dataloader",
    [
        "audyn.utils.data.SequentialBatchDataLoader",
        "audyn.utils.data.DynamicBatchDataLoader",
    ],
)
def test_trainer_for_dataloader(monkeypatch: MonkeyPatch, dataloader: str) -> None:
    """Test BaseTrainer for dataloaders."""
    DATA_SIZE = 20
    BATCH_SIZE = 2
    ITERATIONS = 3

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        monkeypatch.chdir(temp_dir)

        overrides_conf_dir = relpath(join(dirname(realpath(__file__)), "_conf_dummy"), os.getcwd())
        exp_dir = "./exp"

        with hydra.initialize(
            version_base="1.2",
            config_path=relpath(config_template_path, dirname(realpath(__file__))),
            job_name="test_driver",
        ):
            config = hydra.compose(
                config_name="config",
                overrides=create_dummy_for_dataloader_override(
                    overrides_conf_dir=overrides_conf_dir,
                    exp_dir=exp_dir,
                    data_size=DATA_SIZE,
                    batch_size=BATCH_SIZE,
                    iterations=ITERATIONS,
                    dataloader=dataloader,
                ),
                return_hydra_config=True,
            )

        setup_system(config)

        train_dataset = hydra.utils.instantiate(config.train.dataset.train)
        validation_dataset = hydra.utils.instantiate(config.train.dataset.validation)

        train_loader = hydra.utils.instantiate(
            config.train.dataloader.train,
            train_dataset,
            collate_fn=default_collate_fn,
        )
        validation_loader = hydra.utils.instantiate(
            config.train.dataloader.validation,
            validation_dataset,
            collate_fn=default_collate_fn,
        )
        loaders = BaseDataLoaders(train_loader, validation_loader)

        model = instantiate_model(config.model)
        model = set_device(
            model,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )
        optimizer = instantiate_optimizer(config.optimizer, model.parameters())
        lr_scheduler = hydra.utils.instantiate(config.lr_scheduler, optimizer)
        criterion = hydra.utils.instantiate(config.criterion)
        criterion = set_device(
            criterion,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
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


def create_dummy_text_to_feat_override(
    overrides_conf_dir: str,
    exp_dir: str,
    data_size: int,
    batch_size: int = 1,
    iterations: int = 1,
    vocab_size: int = 10,
    n_mels: int = 5,
    up_scale: int = 2,
) -> List[str]:
    sample_rate = 16000
    length = 10

    output_dir, _, tag = exp_dir.rsplit("/", maxsplit=2)
    tensorboard_dir = os.path.join(output_dir, "tensorboard", tag)

    return [
        "train=dummy_text-to-feat",
        "test=dummy",
        "model=dummy_text-to-feat",
        "optimizer=dummy",
        "lr_scheduler=dummy",
        "criterion=dummy",
        f"hydra.searchpath=[{overrides_conf_dir}]",
        "hydra.job.num=1",
        f"hydra.runtime.output_dir={exp_dir}/log",
        f"data.audio.sample_rate={sample_rate}",
        f"train.dataset.train.size={data_size}",
        f"train.dataset.validation.size={data_size}",
        f"train.dataset.train.vocab_size={vocab_size}",
        f"train.dataset.train.n_mels={n_mels}",
        f"train.dataset.train.up_scale={up_scale}",
        f"train.dataset.train.length={length}",
        f"train.dataloader.train.batch_size={batch_size}",
        f"train.dataloader.validation.batch_size={batch_size}",
        f"train.output.exp_dir={exp_dir}",
        f"train.output.tensorboard_dir={tensorboard_dir}",
        f"train.steps.iterations={iterations}",
        f"model.vocab_size={vocab_size}",
        f"model.num_features={n_mels}",
        "criterion.criterion_name.key_mapping.estimated.input=estimated_melspectrogram",
        "criterion.criterion_name.key_mapping.target.target=melspectrogram",
    ]


def create_dummy_feat_to_wave_override(
    overrides_conf_dir: str,
    exp_dir: str,
    data_size: int,
    batch_size: int = 1,
    iterations: int = 1,
    n_mels: int = 5,
    up_scale: int = 2,
) -> List[str]:
    sample_rate = 16000
    length = 10

    output_dir, _, tag = exp_dir.rsplit("/", maxsplit=2)
    tensorboard_dir = os.path.join(output_dir, "tensorboard", tag)

    return [
        "train=dummy_feat-to-wave",
        "test=dummy",
        "model=dummy_feat-to-wave",
        "optimizer=dummy",
        "lr_scheduler=dummy",
        "criterion=dummy",
        f"hydra.searchpath=[{overrides_conf_dir}]",
        "hydra.job.num=1",
        f"hydra.runtime.output_dir={exp_dir}/log",
        f"data.audio.sample_rate={sample_rate}",
        f"train.dataset.train.size={data_size}",
        f"train.dataset.validation.size={data_size}",
        f"train.dataset.train.n_mels={n_mels}",
        f"train.dataset.train.up_scale={up_scale}",
        f"train.dataset.train.length={length}",
        f"train.dataloader.train.batch_size={batch_size}",
        f"train.dataloader.validation.batch_size={batch_size}",
        f"train.output.exp_dir={exp_dir}",
        f"train.output.tensorboard_dir={tensorboard_dir}",
        f"train.steps.iterations={iterations}",
        f"model.num_features={n_mels}",
        "criterion.criterion_name.key_mapping.estimated.input=estimated_waveform",
        "criterion.criterion_name.key_mapping.target.target=waveform",
    ]


def create_dummy_text_to_wave_override(
    overrides_conf_dir: str,
    exp_dir: str,
    data_size: int,
    batch_size: int = 1,
    vocab_size: int = 10,
    up_scale: int = 2,
    text_to_feat_checkpoint: str = None,
    feat_to_wave_checkpoint: str = None,
) -> List[str]:
    sample_rate = 16000
    length = 10

    return [
        "test=dummy_text-to-wave",
        "model=dummy_text-to-wave",
        f"hydra.searchpath=[{overrides_conf_dir}]",
        "hydra.job.num=1",
        f"hydra.runtime.output_dir={exp_dir}/log",
        f"data.audio.sample_rate={sample_rate}",
        f"test.dataset.test.size={data_size}",
        f"test.dataset.test.vocab_size={vocab_size}",
        f"test.dataset.test.up_scale={up_scale}",
        f"test.dataset.test.length={length}",
        f"test.dataloader.test.batch_size={batch_size}",
        f"test.checkpoint.text_to_feat={text_to_feat_checkpoint}",
        f"test.checkpoint.feat_to_wave={feat_to_wave_checkpoint}",
        f"test.output.exp_dir={exp_dir}",
    ]


def create_dummy_for_dataloader_override(
    overrides_conf_dir: str,
    exp_dir: str,
    data_size: int,
    batch_size: int = 1,
    iterations: int = 1,
    dataloader: str = None,
) -> List[str]:
    sample_rate = 16000
    in_channels, out_channels = 3, 2
    kernel_size = 3

    *_, dataloader_name = dataloader.split(".")

    if dataloader_name == "SequentialBatchDataLoader":
        train = "dummy_sequential_dataloader"
        additional_override = [
            f"train.dataloader.train.batch_size={batch_size}",
            f"train.dataloader.validation.batch_size={batch_size}",
        ]
    elif dataloader_name == "DynamicBatchDataLoader":
        train = "dummy_dynamic_dataloader"
        batch_length = 3 * data_size // 4
        additional_override = [
            "train.dataloader.train.key=input",
            "train.dataloader.validation.key=input",
            f"train.dataloader.train.batch_length={batch_length}",
            f"train.dataloader.validation.batch_length={batch_length}",
        ]
    else:
        raise ValueError(f"Invalid dataloader {dataloader_name} is given.")

    override_list = [
        f"train={train}",
        "test=dummy",
        "model=dummy_cnn",
        "optimizer=dummy",
        "lr_scheduler=dummy",
        "criterion=dummy",
        f"hydra.searchpath=[{overrides_conf_dir}]",
        "hydra.job.num=1",
        f"hydra.runtime.output_dir={exp_dir}/log",
        f"data.audio.sample_rate={sample_rate}",
        f"train.dataset.train.size={data_size}",
        f"train.dataset.validation.size={data_size}",
        f"train.dataset.train.num_features={in_channels}",
        f"train.dataset.train.min_length={kernel_size + 1}",
        f"train.dataloader.train._target_={dataloader}",
        f"train.dataloader.validation._target_={dataloader}",
        f"train.output.exp_dir={exp_dir}",
        f"train.steps.iterations={iterations}",
        f"model.in_channels={in_channels}",
        f"model.out_channels={out_channels}",
        f"model.kernel_size={kernel_size}",
    ]

    return override_list + additional_override


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
