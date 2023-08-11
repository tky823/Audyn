import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.2", config_path=None, config_name="config")
def main(config: DictConfig) -> None:
    is_distributed = str(config.system.distributed.enable)
    print(is_distributed.lower())


if __name__ == "__main__":
    main()
