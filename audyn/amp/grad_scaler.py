import torch
from packaging import version

IS_TORCH_LT_2_3 = version.parse(torch.__version__) < version.parse("2.3")

if IS_TORCH_LT_2_3:
    from torch.cuda.amp import GradScaler as _GradScaler
else:
    from torch.amp.grad_scaler import GradScaler as _GradScaler

__all__ = [
    "GradScaler",
]


class GradScaler(_GradScaler):
    """Wrapper class of torch.amp.grad_scaler.GradScaler.

    See torch.amp.grad_scaler.GradScaler for arguments.
    """

    def __init__(self, device="cuda", **kwargs) -> None:
        if device == "mps":
            _device = "cpu"
        else:
            _device = device

        if IS_TORCH_LT_2_3:
            super().__init__(**kwargs)
        else:
            super().__init__(_device, **kwargs)
