import hydra
import torch
from omegaconf import DictConfig


@hydra.main(version_base="1.2", config_path=None, config_name="config")
def main(config: DictConfig) -> None:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        availability = str(config.system.distributed.enable).lower()

        if num_gpus > 1:
            if availability == "false":
                raise ValueError(
                    "Set config.system.distributed.enable=true for multi GPU training."
                )
            else:
                is_distributed = True
        else:
            if availability == "true":
                is_distributed = True
            else:
                is_distributed = False
    else:
        is_distributed = False

    print(str(is_distributed).lower())


if __name__ == "__main__":
    main()
