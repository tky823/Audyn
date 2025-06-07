from omegaconf import DictConfig

import audyn
from audyn.utils import is_available_dump_format


@audyn.main()
def main(config: DictConfig) -> None:
    """Display config.preprocess.dump_format."""
    dump_format = config.preprocess.dump_format

    if is_available_dump_format(dump_format):
        raise ValueError(f"Unknown dump format {dump_format} is detected.")

    print(dump_format)


if __name__ == "__main__":
    main()
