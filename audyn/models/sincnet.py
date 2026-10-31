import warnings
from typing import List, Union

from torch.nn.common_types import _size_1_t

from .pyannote import SincNet as _SincNet


class SincNet(_SincNet):
    """Wrapper class of SincNet."""

    # TODO: implement here for speaker recognition

    def __init__(
        self,
        num_features: List[int],
        kernel_size: Union[List[_size_1_t], _size_1_t],
        pool_size: Union[List[_size_1_t], _size_1_t],
        sample_rate: int = 16000,
        stride: _size_1_t = 10,
    ) -> None:
        warnings.warn(
            "audyn.models.sincnet.SincNet is deprecated and may be removed or "
            "changed in a future release, which could break existing code.",
            category=FutureWarning,
            stacklevel=2,
        )

        super().__init__(
            num_features,
            kernel_size=kernel_size,
            pool_size=pool_size,
            sample_rate=sample_rate,
            stride=stride,
        )
