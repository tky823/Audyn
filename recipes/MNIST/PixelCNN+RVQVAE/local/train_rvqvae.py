import functools
from typing import Dict, Iterable, List, Optional

import torch
from omegaconf import DictConfig

import audyn
from audyn.utils import (
    instantiate,
    instantiate_criterion,
    instantiate_lr_scheduler,
    instantiate_model,
    instantiate_optimizer,
    setup_config,
)
from audyn.utils.data import BaseDataLoaders, default_collate_fn
from audyn.utils.driver import BaseTrainer
from audyn.utils.modules import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    train_dataset = instantiate(config.train.dataset.train)
    validation_dataset = instantiate(config.train.dataset.validation)

    codebook_size = config.data.codebook.size
    num_stages = config.data.codebook.num_stages
    down_scale = config.model.encoder.stride**config.model.encoder.num_layers

    train_loader = instantiate(
        config.train.dataloader.train,
        train_dataset,
        collate_fn=functools.partial(
            collate_fn,
            codebook_size=codebook_size,
            num_stages=num_stages,
            down_scale=down_scale,
        ),
    )
    validation_loader = instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(
            collate_fn,
            codebook_size=codebook_size,
            num_stages=num_stages,
            down_scale=down_scale,
        ),
    )
    loaders = BaseDataLoaders(train_loader, validation_loader)

    model = instantiate_model(config.model)
    model = set_device(
        model,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
    )

    optimizer = instantiate_optimizer(config.optimizer, model)
    lr_scheduler = instantiate_lr_scheduler(config.lr_scheduler, optimizer)

    criterion = instantiate_criterion(config.criterion)
    criterion = set_device(
        criterion,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
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


def collate_fn(
    list_batch: List[Dict[str, torch.Tensor]],
    codebook_size: int,
    num_stages: int,
    down_scale: int,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        list_batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        codebook_size (int): Size of codebook used in VQVAE.
        downscale (int): Scale of downsampling in VQVAE.

    Returns:
        Dict of batch.

    """
    dict_batch = default_collate_fn(list_batch, keys=keys)
    batch_size, _, height, width = dict_batch["input"].size()

    dict_batch["codebook_indices"] = torch.randint(
        0,
        codebook_size,
        (batch_size, num_stages, height // down_scale, width // down_scale),
        dtype=torch.long,
    )

    return dict_batch


if __name__ == "__main__":
    main()
