import torch
from omegaconf import DictConfig

import audyn
from audyn.utils.distributed import is_distributed


@audyn.main()
def main(config: DictConfig) -> None:
    """Determine command to run script.

    If ``config.system`` is distributed,
    ``torchrun --standalone --nnodes=1 --nproc_per_node={nproc_per_node}``
    is returned to stdout. ``torch.cuda.device_count()`` is used as ``nproc_per_node``.
    Otherwise, ``python`` is returned.

    """
    if is_distributed(config.system):
        nproc_per_node = torch.cuda.device_count()

        if config.system.distributed.nodes is None:
            nnodes = 1
        else:
            nnodes = config.system.distributed.nodes

        if nnodes > 1:
            raise NotImplementedError("Only nnodes=1 is supported.")

        cmd = "torchrun"

        if nnodes == 1:
            cmd += " --standalone"

        cmd += f" --nnodes={nnodes} --nproc_per_node={nproc_per_node}"
    else:
        cmd = "python"

    print(cmd)


if __name__ == "__main__":
    main()
