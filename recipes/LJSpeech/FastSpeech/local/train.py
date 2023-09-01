from typing import Dict, Iterable, List, Optional

import hydra
import torch
from omegaconf import DictConfig

import audyn
from audyn.utils import setup_system
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
        collate_fn=collate_fn,
    )
    validation_loader = hydra.utils.instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=collate_fn,
    )
    loaders = BaseDataLoaders(train_loader, validation_loader)

    model = hydra.utils.instantiate(config.model)
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

    trainer = TextToFeatTrainer(
        loaders,
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        config=config,
    )
    trainer.run()


def collate_fn(
    batch: List[Dict[str, torch.Tensor]], keys: Optional[Iterable[str]] = None
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        keys (iterable, optional): Keys to generate batch.
            If ``None`` is given, all keys detected in ``batch`` are used.
            Default: ``None``.

    Returns:
        Dict of batch.

    """
    dict_batch = default_collate_fn(batch, keys=keys)

    if "melspectrogram" in dict_batch.keys():
        dict_batch["melspectrogram"] = dict_batch["melspectrogram"].permute(0, 2, 1)

    return dict_batch


if __name__ == "__main__":
    main()
