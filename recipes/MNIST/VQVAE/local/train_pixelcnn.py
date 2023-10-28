import functools
from typing import Any, Dict, Iterable, List, Optional

import hydra
import torch
from omegaconf import DictConfig

import audyn
from audyn.utils import instantiate_model, setup_system
from audyn.utils.data import BaseDataLoaders, default_collate_fn
from audyn.utils.driver import BaseTrainer
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    train_dataset = hydra.utils.instantiate(config.train.dataset.train)
    validation_dataset = hydra.utils.instantiate(config.train.dataset.validation)

    codebook_size = config.data.codebook.size

    train_loader = hydra.utils.instantiate(
        config.train.dataloader.train,
        train_dataset,
        collate_fn=functools.partial(
            collate_fn,
            codebook_size=codebook_size,
        ),
    )
    validation_loader = hydra.utils.instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(
            collate_fn,
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

    optimizer = hydra.utils.instantiate(config.optimizer, model.parameters())
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


def collate_fn(
    list_batch: List[Dict[str, Any]],
    codebook_size: int,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Generate dict-based batch.

    Args:
        list_batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        codebook_size (int): Size of codebook used in VQVAE.

    Returns:
        Dict of batch.

    """
    dict_batch = default_collate_fn(list_batch, keys=keys)
    batch_size, height, width = dict_batch["indices"].size()
    factory_kwargs = {
        "dtype": dict_batch["indices"].dtype,
        "device": dict_batch["indices"].device,
    }
    dict_batch["initial_index"] = torch.randint(
        0,
        codebook_size,
        (batch_size, 1, 1),
        **factory_kwargs,
    )

    dict_batch["height"] = height
    dict_batch["width"] = width

    return dict_batch


if __name__ == "__main__":
    main()
