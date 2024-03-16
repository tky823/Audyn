from omegaconf import DictConfig
from utils.driver import PriorSaver

import audyn
from audyn.utils import instantiate, instantiate_model, setup_system
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    dataset = instantiate(config.train.dataset)
    loader = instantiate(config.train.dataloader, dataset)

    model = instantiate_model(config.model)
    model = set_device(
        model,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
    )

    saver = PriorSaver(loader, model, config=config)
    saver.run()


if __name__ == "__main__":
    main()
