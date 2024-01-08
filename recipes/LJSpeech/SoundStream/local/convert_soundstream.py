import os

import torch
from omegaconf import DictConfig

import audyn
from audyn.utils import setup_system


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    checkpoint = config.train.checkpoint
    save_path = config.train.output.save_path

    state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    soundstream_generator_config = state_dict["resolved_config"]["model"]["generator"]
    soundstream_generator_config["_target_"] = "utils.models.valle.SoundStreamFirstStageDecoder"
    state_dict["resolved_config"]["model"] = soundstream_generator_config
    soundstream_generator_state_dict = state_dict["model"]["generator"]
    state_dict["model"] = soundstream_generator_state_dict

    save_dir = os.path.dirname(save_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(state_dict, save_path)


if __name__ == "__main__":
    main()
