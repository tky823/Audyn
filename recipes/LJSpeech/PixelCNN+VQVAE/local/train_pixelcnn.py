import functools
from typing import Any, Dict, Iterable, List, Optional

import torch
from omegaconf import DictConfig

import audyn
from audyn.utils import (
    instantiate,
    instantiate_criterion,
    instantiate_grad_clipper,
    instantiate_lr_scheduler,
    instantiate_model,
    instantiate_optimizer,
    setup_config,
)
from audyn.utils.data import BaseDataLoaders, default_collate_fn, slice_feautures
from audyn.utils.driver import BaseTrainer
from audyn.utils.modules import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    train_dataset = instantiate(config.train.dataset.train)
    validation_dataset = instantiate(config.train.dataset.validation)

    codebook_size = config.data.codebook.size

    train_loader = instantiate(
        config.train.dataloader.train,
        train_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            codebook_size=codebook_size,
            random_slice=True,
        ),
    )
    validation_loader = instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            codebook_size=codebook_size,
            random_slice=False,
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
    grad_clipper = instantiate_grad_clipper(config.train.clip_gradient, model.parameters())
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
        grad_clipper=grad_clipper,
        criterion=criterion,
        config=config,
    )
    trainer.run()


def collate_fn(
    list_batch: List[Dict[str, Any]],
    data_config: DictConfig,
    codebook_size: int,
    keys: Optional[Iterable[str]] = None,
    random_slice: bool = True,
) -> Dict[str, Any]:
    """Generate dict-based batch.

    Args:
        list_batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        data_config (DictConfig): Config of data.
        codebook_size (int): Size of codebook used in VQVAE.
        keys (iterable, optional): Keys to generate batch.
            If ``None`` is given, all keys detected in ``batch`` are used.
            Default: ``None``.
        random_slice (bool): If ``random_slice=True``, waveform and
            melspectrogram slices are selected at random. Default: ``True``.

    Returns:
        Dict of batch.

    """
    slice_length = data_config.indices.slice_length

    dict_batch = default_collate_fn(list_batch, keys=keys)

    dict_batch = slice_feautures(
        dict_batch,
        slice_length=slice_length,
        key_mapping={
            "indices": "indices_slice",
        },
        random_slice=random_slice,
    )

    batch_size, height, width = dict_batch["indices_slice"].size()
    factory_kwargs = {
        "dtype": dict_batch["indices_slice"].dtype,
        "device": dict_batch["indices_slice"].device,
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
