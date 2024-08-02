from os.path import dirname, join, realpath
from typing import Any

import hydra

__all__ = ["main"]

config_path = join(dirname(dirname(dirname(realpath(__file__)))), "configs")
config_name = "config"


def main(version_base="1.2", config_path=config_path, config_name=config_name) -> Any:
    """Wrapper function of hydra.main."""
    return hydra.main(version_base=version_base, config_path=config_path, config_name=config_name)
