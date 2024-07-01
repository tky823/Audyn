import functools
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
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
from audyn.utils.driver import BaseTrainer
from audyn.utils.modules import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    dataset_config = config.train.dataset
    dataloader_config = config.train.dataloader
    audio_config = config.data.audio
    melspec_config = config.data.melspectrogram

    train_dataset = instantiate(dataset_config.train)
    validation_dataset = instantiate(dataset_config.validation)

    train_loader = instantiate(
        dataloader_config.train,
        train_dataset,
        collate_fn=functools.partial(
            collate_fn,
            slice_length=audio_config.slice_length,
            random_caption=True,
            random_slice=True,
            freq_mask_param=melspec_config.get("freq_mask_param"),
            time_mask_param=melspec_config.get("time_mask_param"),
        ),
    )
    validation_loader = instantiate(
        dataloader_config.validation,
        validation_dataset,
        collate_fn=functools.partial(
            collate_fn,
            slice_length=audio_config.slice_length,
            random_caption=False,
            random_slice=False,
            freq_mask_param=None,
            time_mask_param=None,
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
    slice_length: Optional[int] = None,
    random_caption: bool = True,
    random_slice: bool = False,
    freq_mask_param: Optional[int] = None,
    time_mask_param: Optional[int] = None,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        random_caption (bool): If ``True``, random caption is used. Otherwise,
            first caption is used, which is useful for validation.
        random_slice (bool): If ``True``, slicing is applied at random poistion.
        freq_mask_param (int, optional): Parameter of frequency masking.
        time_mask_param (int, optional): Parameter of time masking.
        keys (iterable, optional): Keys to generate batch.
            If ``None`` is given, all keys detected in ``batch`` are used.
            Default: ``None``.

    Returns:
        Dict of batch.

    """
    for sample_idx in range(len(batch)):
        sample = batch[sample_idx]
        tokens = sample.pop("tokens")
        tokens, tokens_length = nn.utils.rnn.pad_packed_sequence(tokens, batch_first=True)

        if random_caption:
            caption_idx = torch.randint(0, len(tokens), ()).item()
        else:
            caption_idx = 0

        sample["text"] = tokens[caption_idx]
        sample["text_length"] = tokens_length[caption_idx]

    dict_batch = default_collate_fn(batch, keys=keys)

    if slice_length is not None:
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
            pad_values={
                "melspectrogram": 1,  # log- is padded with 0.
            },
        )

    dict_batch = take_log_features(
        dict_batch,
        key_mapping={
            "melspectrogram_slice": "log_melspectrogram_slice",
        },
    )

    log_melspectrogram_slice = dict_batch["log_melspectrogram_slice"]
    spectrogram_dims = log_melspectrogram_slice.dim()

    # to support torchaudio < 2.1.0
    if spectrogram_dims == 3:
        log_melspectrogram_slice = log_melspectrogram_slice.unsqueeze(dim=-3)

    masking_kwargs = {
        "mask_value": 0.0,
        "p": 1,
    }

    if freq_mask_param is not None:
        log_melspectrogram_slice = aF.mask_along_axis_iid(
            log_melspectrogram_slice,
            freq_mask_param,
            axis=log_melspectrogram_slice.dim() - 2,
            **masking_kwargs,
        )

    if time_mask_param is not None:
        log_melspectrogram_slice = aF.mask_along_axis_iid(
            log_melspectrogram_slice,
            time_mask_param,
            axis=log_melspectrogram_slice.dim() - 1,
            **masking_kwargs,
        )

    if spectrogram_dims == 3:
        log_melspectrogram_slice = log_melspectrogram_slice.squeeze(dim=-3)

    dict_batch["log_melspectrogram_slice"] = log_melspectrogram_slice

    dict_batch.pop("waveform")
    dict_batch.pop("waveform_length")
    dict_batch.pop("melspectrogram")

    return dict_batch


if __name__ == "__main__":
    main()
