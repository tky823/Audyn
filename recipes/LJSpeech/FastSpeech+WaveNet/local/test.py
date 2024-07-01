import functools
from typing import Dict, Iterable, List, Optional

import torch
import torchaudio.functional as aF
from omegaconf import DictConfig

import audyn
from audyn.utils import instantiate, instantiate_cascade_text_to_wave, setup_config
from audyn.utils.data import default_collate_fn
from audyn.utils.driver import CascadeTextToWaveGenerator
from audyn.utils.modules import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    test_dataset = instantiate(config.test.dataset.test)

    test_loader = instantiate(
        config.test.dataloader.test,
        test_dataset,
        collate_fn=functools.partial(collate_fn, data_config=config.data),
    )

    model = instantiate_cascade_text_to_wave(
        config.model,
        text_to_feat_checkpoint=config.test.checkpoint.text_to_feat,
        feat_to_wave_checkpoint=config.test.checkpoint.feat_to_wave,
    )
    model = set_device(
        model,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
    )

    generator = CascadeTextToWaveGenerator(
        test_loader,
        model,
        config=config,
    )
    generator.run()


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    data_config: DictConfig,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        data_config (DictConfig): Config of data.
        keys (iterable, optional): Keys to generate batch.
            If ``None`` is given, all keys detected in ``batch`` are used.
            Default: ``None``.

    Returns:
        Dict of batch.

    """
    quantization_channels = data_config.audio.quantization_channels

    dict_batch = default_collate_fn(batch, keys=keys)
    batch_size = dict_batch["phones"].size(0)

    dict_batch["initial_waveform"] = torch.zeros((batch_size, 1), dtype=torch.float)
    dict_batch["initial_waveform_mulaw"] = aF.mu_law_encoding(
        dict_batch["initial_waveform"], quantization_channels=quantization_channels
    )

    return dict_batch


if __name__ == "__main__":
    main()
