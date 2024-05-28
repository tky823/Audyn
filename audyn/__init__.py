import importlib
import operator
import re
from typing import Any

from omegaconf import OmegaConf

from .utils.hydra import main

__all__ = ["__version__", "main"]

__version__ = "0.0.1.dev7"

# for resolver
_whitespace_re = re.compile(r"\s+")
_int_re = re.compile(r"^\d+$")
_float_re = re.compile(r"^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")


def _constant_resolver(full_var_name: str) -> Any:
    if "+" in full_var_name:
        # TODO: generalize
        # to support whitespace, e.g. ${const:audyn.utils.data.clotho.vocab_size + 1}
        full_var_names = full_var_name.split("+")
        resolved = _resolve(full_var_names[0])

        for _full_var_name in full_var_names[1:]:
            _full_var_name = _whitespace_re.sub("", _full_var_name)

            if _int_re.match(_full_var_name):
                _resolved = int(_full_var_name)
            elif _float_re.match(_full_var_name):
                _resolved = float(_full_var_name)
            else:
                raise ValueError(f"{_full_var_name} cannot be converted to int nor float.")

            resolved = operator.add(resolved, _resolved)
    else:
        resolved = _resolve(full_var_name)

    return resolved


def _resolve(full_var_name: str) -> Any:
    full_var_name = full_var_name.strip()
    mod_name, var_name = full_var_name.rsplit(".", maxsplit=1)

    try:
        resolved = getattr(importlib.import_module(mod_name), var_name)
    except ModuleNotFoundError:
        # TODO: generalize
        attr_name = var_name
        mod_name, var_name = mod_name.rsplit(".", maxsplit=1)
        imported_module = importlib.import_module(mod_name)
        cls = getattr(imported_module, var_name)
        resolved = getattr(cls, attr_name)

    return resolved


OmegaConf.register_new_resolver("const", _constant_resolver)
