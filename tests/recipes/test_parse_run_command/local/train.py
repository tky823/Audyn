from omegaconf import DictConfig

import audyn
from audyn.utils import setup_config


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)


if __name__ == "__main__":
    main()
