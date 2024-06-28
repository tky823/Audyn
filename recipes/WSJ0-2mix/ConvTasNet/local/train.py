from omegaconf import DictConfig

import audyn
from audyn.utils import setup_config
from audyn.utils.driver import AutoTrainer


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    trainer = AutoTrainer.build_from_config(config)
    trainer.run()


if __name__ == "__main__":
    main()
