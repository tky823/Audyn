import importlib
from typing import Any

from omegaconf import OmegaConf

from .utils.hydra import main

__all__ = ["__version__", "main"]

__version__ = "0.0.0"


def _constant_resolver(full_var_name: str) -> Any:
    mod_name, var_name = full_var_name.rsplit(".", maxsplit=1)

    return getattr(importlib.import_module(mod_name), var_name)


OmegaConf.register_new_resolver("const", _constant_resolver)
