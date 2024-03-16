import functools
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig

import audyn
from audyn.utils import instantiate, instantiate_cascade_text_to_wave, setup_system
from audyn.utils.data import default_collate_fn
from audyn.utils.driver import CascadeTextToWaveGenerator
from audyn.utils.model import set_device, unwrap


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    model = instantiate_cascade_text_to_wave(
        config.model,
        text_to_feat_checkpoint=config.test.checkpoint.text_to_feat,
        feat_to_wave_checkpoint=config.test.checkpoint.feat_to_wave,
    )
    model = set_device(
        model,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.test.ddp_kwargs,
    )
    unwrapped_model = unwrap(model)
    down_scale = unwrapped_model.down_scale

    test_dataset = instantiate(config.test.dataset.test)
    test_loader = instantiate(
        config.test.dataloader.test,
        test_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            down_scale=down_scale,
        ),
    )

    generator = CascadeTextToWaveGenerator(
        test_loader,
        model,
        config=config,
    )
    generator.run()


def collate_fn(
    batch: List[Any],
    data_config: DictConfig,
    down_scale: int,
) -> Dict[str, Any]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        data_config (DictConfig): Config of data.
        down_scale (int): Down scale of SoundStream.

    Returns:
        Dict of batch.

    """
    peak_normalization = data_config.audio.peak_normalization

    dict_batch = default_collate_fn(batch)

    dict_batch["waveform"] = dict_batch["waveform"].unsqueeze(dim=1)

    if peak_normalization:
        amplitude = torch.abs(dict_batch["waveform"])
        vmax, _ = torch.max(amplitude, dim=-1)
        vmax = torch.clamp(vmax, min=1e-8)
        dict_batch["waveform"] = dict_batch["waveform"] / vmax.unsqueeze(dim=-1)

    waveform_length = dict_batch["waveform_length"].item()
    max_codebook_indices_length = (waveform_length - 1) // down_scale + 1
    dict_batch["max_codebook_indices_length"] = torch.tensor(
        max_codebook_indices_length, dtype=torch.long
    )

    return dict_batch


if __name__ == "__main__":
    main()
