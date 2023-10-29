import functools
from typing import Dict, Iterable, List, Optional

import hydra
import torch
from omegaconf import DictConfig

import audyn
from audyn.utils import (
    instantiate_lr_scheduler,
    instantiate_model,
    instantiate_optimizer,
    setup_system,
)
from audyn.utils.data import BaseDataLoaders, default_collate_fn
from audyn.utils.driver import BaseTrainer
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    train_dataset = hydra.utils.instantiate(config.train.dataset.train)
    validation_dataset = hydra.utils.instantiate(config.train.dataset.validation)

    down_scale = config.model.encoder.stride**config.model.encoder.num_layers
    codebook_size = config.data.codebook.size

    train_loader = hydra.utils.instantiate(
        config.train.dataloader.train,
        train_dataset,
        collate_fn=functools.partial(
            collate_fn,
            down_scale=down_scale,
            codebook_size=codebook_size,
        ),
    )
    validation_loader = hydra.utils.instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(
            collate_fn,
            down_scale=down_scale,
            codebook_size=codebook_size,
        ),
    )
    loaders = BaseDataLoaders(train_loader, validation_loader)

    model = instantiate_model(config.model)
    model = set_device(
        model,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
    )

    optimizer = instantiate_optimizer(config.optimizer, model)
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


def collate_fn(
    list_batch: List[Dict[str, torch.Tensor]],
    codebook_size: int,
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
        (batch_size, height // down_scale, width // down_scale),
        dtype=torch.long,
    )

    return dict_batch


if __name__ == "__main__":
    main()
