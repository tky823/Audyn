import functools
from typing import Any, Dict, List

import hydra
import torch
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
from audyn.utils.data import BaseDataLoaders, default_collate_fn, slice_feautures
from audyn.utils.driver import BaseTrainer
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    train_dataset = hydra.utils.instantiate(config.train.dataset.train)
    validation_dataset = hydra.utils.instantiate(config.train.dataset.validation)

    down_scale = 1

    for s in config.model.encoder.stride:
        down_scale *= s

    num_layers = config.model.num_layers

    train_loader = hydra.utils.instantiate(
        config.train.dataloader.train,
        train_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            down_scale=down_scale,
            num_layers=num_layers,
            random_slice=True,
        ),
    )
    validation_loader = hydra.utils.instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            down_scale=down_scale,
            num_layers=num_layers,
            random_slice=False,
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
    grad_clipper = instantiate_grad_clipper(config.train.clip_gradient, model.parameters())
    criterion = instantiate_criterion(config.criterion)
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
        grad_clipper=grad_clipper,
        criterion=criterion,
        config=config,
    )
    trainer.run()


def collate_fn(
    batch: List[Any],
    data_config: DictConfig,
    down_scale: int,
    num_layers: int,
    random_slice: bool,
) -> Dict[str, Any]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        data_config (DictConfig): Config of data.
        down_scale (int): Scale of downsampling in RVQ.
        num_layers (int): Number of layers in RVQ.
        random_slice (bool): If ``random_slice=True``, waveform slice is selected at random.
            Default: ``True``.

    Returns:
        Dict of batch.

    """
    dict_batch = default_collate_fn(batch)

    dict_batch["waveform"] = dict_batch["waveform"].unsqueeze(dim=1)
    dict_batch = slice_feautures(
        dict_batch,
        slice_length=data_config.audio.slice_length,
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

    codebook_size = data_config.codebook.size
    batch_size, _, length = dict_batch["waveform"].size()
    dict_batch["codebook_indices"] = torch.randint(
        0,
        codebook_size,
        (batch_size, num_layers, length // down_scale),
        dtype=torch.long,
    )

    return dict_batch


if __name__ == "__main__":
    main()
