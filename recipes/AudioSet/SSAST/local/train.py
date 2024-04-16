from omegaconf import DictConfig

import audyn
from audyn.utils import setup_system
from audyn.utils.driver import BaseTrainer


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    trainer = BaseTrainer.build_from_config(config)
    trainer.run()


if __name__ == "__main__":
    main()
