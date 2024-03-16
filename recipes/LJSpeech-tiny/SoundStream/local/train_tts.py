import functools
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import audyn
from audyn.utils import (
    instantiate,
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

    train_dataset = instantiate(config.train.dataset.train)
    validation_dataset = instantiate(config.train.dataset.validation)

    train_loader = instantiate(
        config.train.dataloader.train,
        train_dataset,
        collate_fn=functools.partial(collate_fn, data_config=config.data),
    )
    validation_loader = instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(collate_fn, data_config=config.data),
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
    grad_clipper = instantiate_grad_clipper(config.train.clip_gradient, model.parameters())
    criterion = instantiate_criterion(config.criterion)
    criterion = set_device(
        criterion,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
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


def collate_fn(batch: List[Any], data_config: DictConfig) -> Dict[str, Any]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        data_config (DictConfig): Config of data.

    Returns:
        Dict of batch.

    """
    pad_idx = data_config.codebook.pad_idx
    eos_idx = data_config.codebook.eos_idx

    assert pad_idx == 0
    assert eos_idx == 1

    list_batch = []

    for sample in batch:
        codebook_indices = sample["codebook_indices"]

        # extract first stage
        codebook_indices, _ = torch.split(
            codebook_indices, [1, codebook_indices.size(0) - 1], dim=0
        )
        codebook_indices = codebook_indices.squeeze(dim=0)

        # trim codebook by random length to simulate variable-length input
        codebook_length = torch.randint(
            codebook_indices.size(-1) // 2, codebook_indices.size(-1) + 1, (), dtype=torch.long
        )
        codebook_length = codebook_length.item()
        codebook_indices, _ = torch.split(
            codebook_indices,
            [codebook_length, codebook_indices.size(-1) - codebook_length],
            dim=-1,
        )

        # shift indices for padding & eos token
        codebook_indices = codebook_indices + 2

        # insert eos index
        codebook_indices = F.pad(codebook_indices, (0, 1), value=eos_idx)

        sample["codebook_indices"] = codebook_indices
        sample["codebook_indices_length"] = torch.tensor(
            codebook_indices.size(-1), dtype=torch.long
        )
        list_batch.append(sample)

    dict_batch = default_collate_fn(list_batch)
    dict_batch["max_codebook_indices_length"] = torch.max(
        dict_batch["codebook_indices_length"]
    ).item()

    return dict_batch


if __name__ == "__main__":
    main()
