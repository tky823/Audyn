import functools
from typing import Dict, Iterable, List, Optional

import hydra
import torch
from omegaconf import DictConfig

import audyn
from audyn.criterion.gan import GANCriterion
from audyn.models.gan import BaseGAN
from audyn.optim.lr_scheduler import GANLRScheduler
from audyn.optim.optimizer import GANOptimizer
from audyn.utils import instantiate_grad_clipper, instantiate_model, setup_system
from audyn.utils.clip_grad import GANGradClipper
from audyn.utils.data import (
    BaseDataLoaders,
    default_collate_fn,
    slice_feautures,
    take_log_features,
)
from audyn.utils.driver import GANTrainer
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    train_dataset = hydra.utils.instantiate(config.train.dataset.train)
    validation_dataset = hydra.utils.instantiate(config.train.dataset.validation)

    train_loader = hydra.utils.instantiate(
        config.train.dataloader.train,
        train_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            random_slice=True,
        ),
    )
    validation_loader = hydra.utils.instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            random_slice=False,
        ),
    )
    loaders = BaseDataLoaders(train_loader, validation_loader)

    generator = instantiate_model(config.model.generator)
    generator = set_device(
        generator,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
    )
    generator_optimizer = hydra.utils.instantiate(
        config.optimizer.generator, generator.parameters()
    )
    generator_lr_scheduler = hydra.utils.instantiate(
        config.lr_scheduler.generator, generator_optimizer
    )
    generator_grad_clipper = instantiate_grad_clipper(
        config.clip_gradient.generator, generator.parameters()
    )
    generator_criterion = hydra.utils.instantiate(config.criterion.generator)
    generator_criterion = set_device(
        generator_criterion,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
    )

    discriminator = instantiate_model(config.model.discriminator)
    discriminator = set_device(
        discriminator,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
    )
    discriminator_optimizer = hydra.utils.instantiate(
        config.optimizer.discriminator, discriminator.parameters()
    )
    discriminator_lr_scheduler = hydra.utils.instantiate(
        config.lr_scheduler.discriminator, discriminator_optimizer
    )
    discriminator_grad_clipper = instantiate_grad_clipper(
        config.clip_gradient.discriminator, discriminator.parameters()
    )
    discriminator_criterion = hydra.utils.instantiate(config.criterion.discriminator)
    discriminator_criterion = set_device(
        discriminator_criterion,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
    )

    model = BaseGAN(generator, discriminator)
    optimizer = GANOptimizer(generator_optimizer, discriminator_optimizer)
    lr_scheduler = GANLRScheduler(generator_lr_scheduler, discriminator_lr_scheduler)
    grad_clipper = GANGradClipper(generator_grad_clipper, discriminator_grad_clipper)
    criterion = GANCriterion(generator_criterion, discriminator_criterion)

    trainer = GANTrainer(
        loaders,
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
        grad_clipper=grad_clipper,
        criterion=criterion,
        config=config,
    )
    trainer.run()


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    data_config: DictConfig,
    keys: Optional[Iterable[str]] = None,
    random_slice: bool = True,
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        data_config (DictConfig): Config of data.
        keys (iterable, optional): Keys to generate batch.
            If ``None`` is given, all keys detected in ``batch`` are used.
            Default: ``None``.
        random_slice (bool): If ``random_slice=True``, waveform and
            melspectrogram slices are selected at random. Default: ``True``.

    Returns:
        Dict of batch.

    """
    hop_length = data_config.melspectrogram.hop_length
    slice_length = data_config.audio.slice_length

    dict_batch = default_collate_fn(batch, keys=keys)

    dict_batch["waveform"] = dict_batch["waveform"].unsqueeze(dim=1)

    dict_batch = slice_feautures(
        dict_batch,
        slice_length=slice_length,
        key_mapping={
            "waveform": "waveform_slice",
            "melspectrogram": "melspectrogram_slice",
        },
        hop_lengths={
            "waveform": 1,
            "melspectrogram": hop_length,
        },
        length_mapping={
            "waveform": "waveform_length",
            "melspectrogram": "melspectrogram_length",
        },
        random_slice=random_slice,
    )
    dict_batch = take_log_features(
        dict_batch,
        key_mapping={
            "melspectrogram": "log_melspectrogram",
            "melspectrogram_slice": "log_melspectrogram_slice",
        },
        flooring_fn=lambda x: torch.clamp(x, min=1e-10),
    )
    dict_batch["max_waveform_slice_length"] = dict_batch["waveform_slice"].size(-1)

    return dict_batch


if __name__ == "__main__":
    main()
