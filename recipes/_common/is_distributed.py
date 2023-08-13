from omegaconf import DictConfig

import audyn
from audyn.utils.distributed import is_distributed


@audyn.main()
def main(config: DictConfig) -> None:
    print(str(is_distributed(config.system)).lower())


if __name__ == "__main__":
    main()
