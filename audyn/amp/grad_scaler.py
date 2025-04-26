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

        super().__init__(_device, **kwargs)
