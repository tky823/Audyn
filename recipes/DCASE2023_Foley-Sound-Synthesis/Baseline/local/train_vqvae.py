import functools
from typing import Any, Dict, Iterable, List, Optional

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import audyn
from audyn.utils import (
    instantiate_criterion,
    instantiate_lr_scheduler,
    instantiate_model,
    instantiate_optimizer,
    setup_system,
)
from audyn.utils.data import BaseDataLoaders, default_collate_fn, take_log_features
from audyn.utils.driver import TextToFeatTrainer
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
            codebook_size=codebook_size,
            down_scale=down_scale,
        ),
    )
    validation_loader = hydra.utils.instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(
            collate_fn,
            codebook_size=codebook_size,
            down_scale=down_scale,
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
        criterion=criterion,
        config=config,
    )
    trainer.run()


def collate_fn(
    list_batch: List[Dict[str, Any]],
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
        keys (iterable, optional): Keys to generate batch.
            If ``None`` is given, all keys detected in ``batch`` are used.
            Default: ``None``.

    Returns:
        Dict of batch.

    """
    dict_batch = default_collate_fn(list_batch, keys=keys)

    # n_mels and n_frames should be divisible by down_scale.
    batch_size, n_mels, n_frames = dict_batch["melspectrogram"].size()
    dict_batch["melspectrogram"] = F.pad(
        dict_batch["melspectrogram"], (0, -(n_frames % down_scale), 0, -(n_mels % down_scale))
    )
    n_mels = n_mels - (n_mels % down_scale)
    n_frames = n_frames - (n_frames % down_scale)

    dict_batch = take_log_features(
        dict_batch,
        key_mapping={
            "melspectrogram": "log_melspectrogram",
        },
        flooring_fn=lambda x: torch.clamp(x, min=1e-10),
    )

    dict_batch["codebook_indices"] = torch.randint(
        0,
        codebook_size,
        (batch_size, n_mels // down_scale, n_frames // down_scale),
        dtype=torch.long,
    )

    return dict_batch


if __name__ == "__main__":
    main()
