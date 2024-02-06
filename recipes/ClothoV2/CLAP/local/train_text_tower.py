import functools
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
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
from audyn.utils.driver import BaseTrainer
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    train_dataset = instantiate(config.train.dataset.train)
    validation_dataset = instantiate(config.train.dataset.validation)

    text_config = config.data.text
    text_preprocessor = instantiate(text_config.preprocessor)

    collate_fn_kwargs = {
        "vocab_size": text_preprocessor.vocab_size,
        "mask_index": text_preprocessor.mask_index,
        "selection_rate": text_config.selection_rate,
        "mask_rate": text_config.mask_rate,
        "replace_rate": text_config.replace_rate,
    }

    train_loader = instantiate(
        config.train.dataloader.train,
        train_dataset,
        collate_fn=functools.partial(
            collate_fn,
            random_caption=True,
            **collate_fn_kwargs,
        ),
    )
    validation_loader = instantiate(
        config.train.dataloader.validation,
        validation_dataset,
        collate_fn=functools.partial(
            collate_fn,
            random_caption=False,
            **collate_fn_kwargs,
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
    random_caption: bool = True,
    vocab_size: int = None,
    mask_index: int = None,
    selection_rate: float = 0.15,
    mask_rate: float = 0.8,
    replace_rate: float = 0.1,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        random_caption (bool): If ``True``, random caption is used. Otherwise,
            first caption is used, which is useful for validation.
        vocab_size: Vocabulary size.
        mask_index: Index of <MASK> token.
        selection_rate (float): Selection probability to mask or replace.
        mask_rate (float): Masking probability of selected tokens.
        replace_rate (float): Replacement probability of selected tokens.
        keys (iterable, optional): Keys to generate batch.
            If ``None`` is given, all keys detected in ``batch`` are used.
            Default: ``None``.

    Returns:
        Dict of batch.

    """
    if vocab_size is None:
        raise ValueError("Set vocab_size.")

    if mask_index is None:
        raise ValueError("Set mask_index.")

    assert 0 <= selection_rate <= 1
    assert 0 <= mask_rate <= 1
    assert 0 <= replace_rate <= 1
    assert 0 <= mask_rate + replace_rate <= 1

    for sample_idx in range(len(batch)):
        sample = batch[sample_idx]
        tokens = sample.pop("tokens")
        tokens, tokens_length = nn.utils.rnn.pad_packed_sequence(tokens, batch_first=True)

        if random_caption:
            caption_idx = torch.randint(0, len(tokens), ()).item()
        else:
            caption_idx = 0

        text = tokens[caption_idx]
        text_length = tokens_length[caption_idx]

        device = text.device

        rand_value = torch.rand(text.size())
        selection_mask = rand_value < selection_rate
        rand_value = torch.rand(text.size())
        masking_mask = selection_mask & (rand_value < mask_rate)
        replacement_mask = selection_mask & ((1 - rand_value) < replace_rate)
        replacement_index = torch.randint(0, vocab_size, text.size())

        masking_mask = masking_mask.to(device)
        replacement_mask = replacement_mask.to(device)
        replacement_index = replacement_index.to(device)

        masked_text = torch.where(masking_mask, mask_index, text)
        masked_text = torch.where(replacement_mask, replacement_index, masked_text)

        sample["text"] = text
        sample["masked_text"] = masked_text
        sample["text_length"] = text_length

    dict_batch = default_collate_fn(batch, keys=keys)
    dict_batch.pop("waveform")
    dict_batch.pop("waveform_length")
    dict_batch.pop("melspectrogram")
    dict_batch.pop("melspectrogram_length")

    return dict_batch


if __name__ == "__main__":
    main()
