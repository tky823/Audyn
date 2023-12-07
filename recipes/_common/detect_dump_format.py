from omegaconf import DictConfig

import audyn
from audyn.utils.data.dataset import available_dump_formats


@audyn.main()
def main(config: DictConfig) -> None:
    """Display config.preprocess.dump_format."""
    dump_format = config.preprocess.dump_format

    if dump_format not in available_dump_formats:
        raise ValueError(f"Unknown dump format {dump_format} is detected.")

    print(dump_format)


if __name__ == "__main__":
    main()
