import functools
from typing import Any, Dict, List

import torch
import torchaudio.functional as aF
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
from audyn.utils.driver import FeatToWaveTrainer
from audyn.utils.modules import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    train_dataset = instantiate(config.train.dataset.train)
    validation_dataset = instantiate(config.train.dataset.validation)

    train_loader = instantiate(
        config.train.dataloader.train,
        train_dataset,
        collate_fn=functools.partial(collate_fn, data_config=config.data, random_slice=True),
    )
    validation_loader = instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(collate_fn, data_config=config.data, random_slice=False),
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

    trainer = FeatToWaveTrainer(
        loaders,
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
        grad_clipper=grad_clipper,
        criterion=criterion,
        config=config,
    )
    trainer.run()


def collate_fn(batch: List[Any], data_config: DictConfig, random_slice: bool) -> Dict[str, Any]:
    quantization_channels = data_config.audio.quantization_channels

    dict_batch = default_collate_fn(batch)

    batch_size_keys = sorted(list(dict_batch.keys()))
    batch_size_key = batch_size_keys[0]
    batch_size = len(dict_batch[batch_size_key])

    dict_batch = slice_feautures(
        dict_batch,
        slice_length=data_config.audio.slice_length,
        key_mapping={
            "waveform": "waveform_slice",
            "waveform_mulaw": "waveform_slice_mulaw",
            "melspectrogram": "melspectrogram_slice",
        },
        hop_lengths={
            "waveform": 1,
            "waveform_mulaw": 1,
            "melspectrogram": data_config.melspectrogram.hop_length,
        },
        length_mapping={
            "waveform": "waveform_length",
            "waveform_mulaw": "waveform_length",
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

    dict_batch["initial_waveform"] = torch.zeros((batch_size, 1), dtype=torch.float)
    dict_batch["initial_waveform_mulaw"] = aF.mu_law_encoding(
        dict_batch["initial_waveform"],
        quantization_channels=quantization_channels,
    )
    dict_batch["max_waveform_slice_length"] = dict_batch["waveform_slice"].size(-1)

    return dict_batch


if __name__ == "__main__":
    main()
