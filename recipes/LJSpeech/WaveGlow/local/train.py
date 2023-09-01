import functools
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig

import audyn
from audyn.utils import setup_system
from audyn.utils.data import (
    BaseDataLoaders,
    default_collate_fn,
    make_noise,
    slice_feautures,
    take_log_features,
)
from audyn.utils.driver import FeatToWaveTrainer
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    train_dataset = hydra.utils.instantiate(config.train.dataset.train)
    validation_dataset = hydra.utils.instantiate(config.train.dataset.validation)

    train_loader = hydra.utils.instantiate(
        config.train.dataloader.train,
        train_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            random_slice=True,
            std=config.data.noise_std.train,
        ),
    )
    validation_loader = hydra.utils.instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            random_slice=False,
            std=config.data.noise_std.validation,
        ),
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

    trainer = FeatToWaveTrainer(
        loaders,
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        config=config,
    )
    trainer.run()


def collate_fn(
    batch: List[Any],
    data_config: DictConfig,
    random_slice: bool,
    std: float = 1,
) -> Dict[str, Any]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        data_config (DictConfig): Config of data.
        random_slice (bool): If ``random_slice=True``, waveform and
            melspectrogram slices are selected at random. Default: ``True``.
        std (float): Standard deviation of noise.

    Returns:
        Dict of batch.

    """
    dict_batch = default_collate_fn(batch)

    batch_size_keys = sorted(list(dict_batch.keys()))
    batch_size_key = batch_size_keys[0]
    batch_size = len(dict_batch[batch_size_key])

    dict_batch["waveform"] = dict_batch["waveform"].unsqueeze(dim=1)
    dict_batch = slice_feautures(
        dict_batch,
        slice_length=data_config.slice_length,
        key_mapping={
            "waveform": "waveform_slice",
            "melspectrogram": "melspectrogram_slice",
        },
        hop_lengths={
            "waveform": 1,
            "melspectrogram": data_config.melspectrogram.hop_length,
        },
        length_mapping={
            "waveform": "waveform_length",
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
    dict_batch = make_noise(
        dict_batch,
        key_mapping={
            "waveform": "noise",
        },
        std=std,
    )

    dict_batch["zeros"] = torch.zeros((batch_size,), dtype=torch.float)
    dict_batch["max_waveform_slice_length"] = dict_batch["waveform_slice"].size(-1)

    return dict_batch


if __name__ == "__main__":
    main()
