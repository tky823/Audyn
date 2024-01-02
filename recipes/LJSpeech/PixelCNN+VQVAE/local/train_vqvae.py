import functools
from typing import Any, Dict, Iterable, List, Optional

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
from audyn.utils.data import (
    BaseDataLoaders,
    default_collate_fn,
    slice_feautures,
    take_log_features,
)
from audyn.utils.driver import BaseTrainer
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    train_dataset = hydra.utils.instantiate(config.train.dataset.train)
    validation_dataset = hydra.utils.instantiate(config.train.dataset.validation)

    codebook_size = config.data.codebook.size
    down_scale = config.model.encoder.stride**config.model.encoder.num_stacks

    train_loader = hydra.utils.instantiate(
        config.train.dataloader.train,
        train_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            codebook_size=codebook_size,
            down_scale=down_scale,
            random_slice=True,
        ),
    )
    validation_loader = hydra.utils.instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            codebook_size=codebook_size,
            down_scale=down_scale,
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
    down_scale: int,
    keys: Optional[Iterable[str]] = None,
    random_slice: bool = True,
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        list_batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        data_config (DictConfig): Config of data.
        codebook_size (int): Size of codebook used in VQVAE.
        down_scale (int): Scale of downsampling in VQVAE.
        keys (iterable, optional): Keys to generate batch.
            If ``None`` is given, all keys detected in ``batch`` are used.
            Default: ``None``.
        random_slice (bool): If ``random_slice=True``, waveform and
            melspectrogram slices are selected at random. Default: ``True``.

    Returns:
        Dict of batch.

    """
    slice_length = data_config.audio.slice_length

    dict_batch = default_collate_fn(list_batch, keys=keys)

    dict_batch = slice_feautures(
        dict_batch,
        slice_length=slice_length,
        key_mapping={
            "melspectrogram": "melspectrogram_slice",
        },
        length_mapping={
            "melspectrogram": "melspectrogram_length",
        },
        random_slice=random_slice,
    )
    dict_batch = take_log_features(
        dict_batch,
        key_mapping={
            "melspectrogram": "log_melspectrogram",
            "melspectrogram_slice": "log_melspectrogram_slice",
        },
    )

    batch_size, n_mels, n_frames = dict_batch["melspectrogram_slice"].size()

    dict_batch["codebook_indices"] = torch.randint(
        0,
        codebook_size,
        (batch_size, n_mels // down_scale, n_frames // down_scale),
        dtype=torch.long,
    )

    return dict_batch


if __name__ == "__main__":
    main()
