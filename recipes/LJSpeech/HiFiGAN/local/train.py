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
from audyn.utils.data import BaseDataLoaders
from audyn.utils.driver import GANTrainer
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    train_dataset = instantiate(config.train.dataset.train)
    validation_dataset = instantiate(config.train.dataset.validation)

    train_loader = instantiate(
        config.train.dataloader.train,
        train_dataset,
    )
    validation_loader = instantiate(
        config.train.dataloader.validation,
        validation_dataset,
    )
    loaders = BaseDataLoaders(train_loader, validation_loader)

    generator = instantiate_gan_generator(config.model)
    generator = set_device(
        generator,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
    )
    generator_optimizer = instantiate_optimizer(config.optimizer.generator, generator.parameters())
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

    discriminator = instantiate_gan_discriminator(config.model)
    discriminator = set_device(
        discriminator,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
    )
    discriminator_optimizer = instantiate_optimizer(
        config.optimizer.discriminator, discriminator.parameters()
    )
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


if __name__ == "__main__":
    main()
