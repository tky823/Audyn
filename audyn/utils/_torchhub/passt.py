from typing import Optional

import torch.nn as nn

from ...models.passt import PaSST


def passt_base(
    stride: int = 10,
    patchout: str = "struct",
    n_bins: Optional[int] = None,
    n_frames: Optional[int] = None,
    aggregator: Optional[nn.Module] = None,
    head: Optional[nn.Module] = None,
) -> PaSST:
    """Build SelfSupervisedAudioSpectrogramTransformer.

    Args:
        stride (int): Stride of patch.
        patchout (str): Type of patchout technique.
        aggregator (nn.Module, optional): Aggregator module.
        head (nn.Module, optional): Head module.

    """
    if stride == 10 and patchout == "struct":
        pretrained_model_name = f"passt-base-stride{stride}-{patchout}-ap0.476-swa"
    else:
        raise ValueError(f"Model satisfying stride={stride} and patchout={patchout} is not found.")

    model = PaSST.build_from_pretrained(
        pretrained_model_name,
        stride=stride,
        n_bins=n_bins,
        n_frames=n_frames,
        aggregator=aggregator,
        head=head,
    )

    return model
