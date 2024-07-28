import torch

__all__ = [
    "is_gpu_supported",
]


def is_gpu_supported() -> bool:
    """Judge availability of ``torch.compile`` by checking device capability.

    Returns:
        bool: Whether ``torch.compile`` is available.

    """
    device_capability = torch.cuda.get_device_capability()

    if device_capability in [(7, 0), (8, 0), (9, 0)]:
        return True
    else:
        return False
