import hydra
from omegaconf import DictConfig

import audyn
from audyn.utils import (
    instantiate_criterion,
    instantiate_grad_clipper,
    instantiate_lr_scheduler,
    instantiate_model,
    instantiate_optimizer,
    setup_system,
)
from audyn.utils.data import BaseDataLoaders, default_collate_fn
from audyn.utils.driver import TextToFeatTrainer
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
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

    if hasattr(config.train, "clip_gradient"):
        grad_clipper = instantiate_grad_clipper(config.train.clip_gradient, model.parameters())
    else:
        grad_clipper = None

    criterion = instantiate_criterion(config.criterion)
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
        grad_clipper=grad_clipper,
        criterion=criterion,
        config=config,
    )
    trainer.run()


if __name__ == "__main__":
    main()
