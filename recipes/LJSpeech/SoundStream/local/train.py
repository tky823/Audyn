import functools
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig

import audyn
from audyn.criterion.gan import GANCriterion
from audyn.models.gan import BaseGAN
from audyn.optim.lr_scheduler import GANLRScheduler
from audyn.optim.optimizer import GANOptimizer
from audyn.utils import (
    instantiate,
    instantiate_criterion,
    instantiate_gan_discriminator,
    instantiate_gan_generator,
    instantiate_grad_clipper,
    instantiate_lr_scheduler,
    instantiate_optimizer,
    setup_config,
)
from audyn.utils.clip_grad import GANGradClipper
from audyn.utils.data import BaseDataLoaders, default_collate_fn, slice_feautures
from audyn.utils.driver import GANTrainer
from audyn.utils.modules import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    train_dataset = instantiate(config.train.dataset.train)
    validation_dataset = instantiate(config.train.dataset.validation)

    down_scale = 1

    for s in config.model.generator.encoder.stride:
        down_scale *= s

    num_stages = config.model.generator.num_stages

    train_loader = instantiate(
        config.train.dataloader.train,
        train_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            down_scale=down_scale,
            num_stages=num_stages,
            random_slice=True,
        ),
    )
    validation_loader = instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            down_scale=down_scale,
            num_stages=num_stages,
            random_slice=False,
        ),
    )
    loaders = BaseDataLoaders(train_loader, validation_loader)

    # generator
    generator = instantiate_gan_generator(config.model)
    generator = set_device(
        generator,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
    )
    generator_optimizer = instantiate_optimizer(config.optimizer.generator, generator)
    generator_lr_scheduler = instantiate_lr_scheduler(
        config.lr_scheduler.generator, generator_optimizer
    )
    generator_grad_clipper = instantiate_grad_clipper(
        config.train.clip_gradient.generator, generator.parameters()
    )
    generator_criterion = instantiate_criterion(config.criterion.generator)
    generator_criterion = set_device(
        generator_criterion,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
    )

    # discriminator
    discriminator = instantiate_gan_discriminator(config.model)
    discriminator = set_device(
        discriminator,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
    )
    discriminator_optimizer = instantiate_optimizer(config.optimizer.discriminator, discriminator)
    discriminator_lr_scheduler = instantiate_lr_scheduler(
        config.lr_scheduler.discriminator, discriminator_optimizer
    )
    discriminator_grad_clipper = instantiate_grad_clipper(
        config.train.clip_gradient.discriminator, discriminator.parameters()
    )
    discriminator_criterion = instantiate_criterion(config.criterion.discriminator)
    discriminator_criterion = set_device(
        discriminator_criterion,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
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
    batch: List[Any],
    data_config: DictConfig,
    down_scale: int,
    num_stages: int,
    random_slice: bool,
) -> Dict[str, Any]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        data_config (DictConfig): Config of data.
        down_scale (int): Scale of downsampling in RVQ.
        num_stages (int): Number of stages in RVQ.
        random_slice (bool): If ``random_slice=True``, waveform slice is selected at random.
            Default: ``True``.

    Returns:
        Dict of batch.

    """
    slice_length = data_config.audio.slice_length
    codebook_size = data_config.codebook.size
    peak_normalization = data_config.audio.peak_normalization

    dict_batch = default_collate_fn(batch)

    dict_batch["waveform"] = dict_batch["waveform"].unsqueeze(dim=1)
    dict_batch = slice_feautures(
        dict_batch,
        slice_length=slice_length,
        key_mapping={
            "waveform": "waveform_slice",
        },
        hop_lengths={
            "waveform": 1,
        },
        length_mapping={
            "waveform": "waveform_length",
        },
        random_slice=random_slice,
    )

    if peak_normalization:
        amplitude = torch.abs(dict_batch["waveform_slice"])
        vmax, _ = torch.max(amplitude, dim=-1)
        vmax = torch.clamp(vmax, min=1e-8)
        dict_batch["waveform_slice"] = dict_batch["waveform_slice"] / vmax.unsqueeze(dim=-1)

    batch_size, _, length = dict_batch["waveform"].size()
    dict_batch["codebook_indices"] = torch.randint(
        0,
        codebook_size,
        (batch_size, num_stages, length // down_scale),
        dtype=torch.long,
    )

    return dict_batch


if __name__ == "__main__":
    main()
