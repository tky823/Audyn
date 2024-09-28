import sys

import torch
from packaging import version

IS_WINDOWS = sys.platform == "win32"
IS_PYTHON_GE_3_11 = sys.version_info >= (3, 11)
IS_TORCH_LT_2_0 = version.parse(torch.__version__) < version.parse("2.0")
IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")

__all__ = [
    "is_supported",
    "is_gpu_supported",
]


def is_supported() -> bool:
    """Judge availability of ``torch.compile``.

    Returns:
        bool: Whether ``torch.compile`` is available.

    """
    if IS_WINDOWS:
        return False

    if IS_TORCH_LT_2_0:
        return False

    if IS_TORCH_LT_2_1 and IS_PYTHON_GE_3_11:
        return False

    return True


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
