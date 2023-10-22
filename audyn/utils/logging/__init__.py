import os
from logging import Logger, getLogger
from typing import Optional

try:
    from torch.distributed.elastic.utils.logging import _derive_module_name
except ImportError:

    def _derive_module_name(*args, **kwargs) -> str:
        return "Logger"


__all__ = ["get_logger"]


def get_logger(name: Optional[str] = None, is_distributed: bool = False) -> Logger:
    """Get logger by name.

    When ``is_distributed=True`` and ``int(os.environ["RANK"])>0``,
    a dummy logger is returned. The dummy logger ignores INFO level logs.

    Args:
        name (str, optional): Name of the logger. If no name provided, the name will
              be derived from the call stack.

    Returns:
        Logger: Logger to record process.

    """

    if name is None:
        name = _derive_module_name(depth=2)

    logger = _setup_logger(name, is_distributed=is_distributed)

    return logger


def _setup_logger(name: Optional[str] = None, is_distributed: bool = False):
    if is_distributed and int(os.environ["RANK"]) > 0:
        logger = DummyLogger(name)
    else:
        logger = getLogger(name)

    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    return logger


class DummyLogger(Logger):
    def __init__(self, name: str, level: int = 0) -> None:
        super().__init__(name, level=level)

    def info(self, *args, **kwargs) -> None:
        pass
