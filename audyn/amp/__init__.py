import torch
from torch.amp import autocast as _autocast

__all__ = [
    "autocast",
    "get_autocast_device_type",
]


class autocast(_autocast):
    """Wrapper class of torch.amp.autocast."""

    def __init__(
        self,
        device_type: str,
        enabled: bool = True,
        dtype: torch.dtype = None,
        cache_enabled: bool = True,
    ) -> None:
        if device_type == "mps":
            _device_type = "cpu"
        else:
            _device_type = device_type

        super().__init__(
            device_type=_device_type,
            enabled=enabled,
            dtype=dtype,
            cache_enabled=cache_enabled,
        )


def get_autocast_device_type(*args) -> str:
    if len(args) == 0:
        if torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"
    elif len(args) == 1:
        tensor = args[0]

        assert isinstance(tensor, torch.Tensor)

        if tensor.is_cuda:
            device_type = "cuda"
        else:
            device_type = "cpu"
    else:
        raise RuntimeError("Invalid length of argument is given to get_autocast_device_type.")

    return device_type
