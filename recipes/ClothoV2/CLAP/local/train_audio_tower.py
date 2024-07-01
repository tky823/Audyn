import functools
from typing import Dict, Iterable, List, Optional

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
from audyn.utils.data import (
    BaseDataLoaders,
    default_collate_fn,
    slice_feautures,
    take_log_features,
)
from audyn.utils.driver import BaseTrainer
from audyn.utils.modules import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    train_dataset = instantiate(config.train.dataset.train)
    validation_dataset = instantiate(config.train.dataset.validation)

    slice_length = config.data.audio.slice_length

    train_loader = instantiate(
        config.train.dataloader.train,
        train_dataset,
        collate_fn=functools.partial(
            collate_fn,
            slice_length=slice_length,
            random_slice=True,
        ),
    )
    validation_loader = instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(
            collate_fn,
            slice_length=slice_length,
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
    batch: List[Dict[str, torch.Tensor]],
    slice_length: int,
    random_slice: bool = False,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        slice_length (int): Slice length of Mel-spectrogram.
        random_slice (bool): If ``True``, slicing is applied at random poistion.
        keys (iterable, optional): Keys to generate batch.
            If ``None`` is given, all keys detected in ``batch`` are used.
            Default: ``None``.

    Returns:
        Dict of batch.

    """

    def flooring_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=1e-10)

    for sample_idx in range(len(batch)):
        sample = batch[sample_idx]
        _ = sample.pop("tokens")

    dict_batch = default_collate_fn(batch, keys=keys)

    dict_batch = slice_feautures(
        dict_batch,
        slice_length=slice_length,
        key_mapping={
            "melspectrogram": "melspectrogram_slice",
        },
        hop_lengths={
            "melspectrogram": 1,
        },
        length_mapping={
            "melspectrogram": "melspectrogram_length",
        },
        random_slice=random_slice,
    )
    dict_batch = take_log_features(
        dict_batch,
        key_mapping={
            "melspectrogram_slice": "log_melspectrogram_slice",
        },
        flooring_fn=flooring_fn,
    )

    dict_batch.pop("waveform")
    dict_batch.pop("waveform_length")
    dict_batch.pop("melspectrogram")
    dict_batch.pop("melspectrogram_length")

    return dict_batch


if __name__ == "__main__":
    main()
