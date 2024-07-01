import warnings

import torch
from omegaconf import DictConfig

import audyn
from audyn.utils.distributed import is_distributed


@audyn.main()
def main(config: DictConfig) -> None:
    """Determine command to run script.

    If ``config.system`` is distributed, ``torchrun --nnodes=...`` is returned to stdout.
    Otherwise, ``python`` is returned.

    """
    if is_distributed(config.system):
        distributed_config = config.system.distributed

        if distributed_config.nodes is None:
            nnodes = 1
        else:
            nnodes = distributed_config.nodes

        if distributed_config.nproc_per_node is None:
            nproc_per_node = torch.cuda.device_count()
        else:
            nproc_per_node = distributed_config.nproc_per_node

        cmd = "torchrun"

        if nnodes > 1:
            warnings.warn(
                "Support of multi-node training is limited.",
                UserWarning,
                stacklevel=2,
            )
            rdzv_id = distributed_config.rdzv_id
            rdzv_backend = distributed_config.rdzv_backend
            rdzv_endpoint = distributed_config.rdzv_endpoint
            max_restarts = distributed_config.max_restarts

            if rdzv_id is not None:
                cmd += f" --rdzv-id={rdzv_id}"

            if rdzv_backend is None:
                cmd += f" --rdzv-backend={rdzv_backend}"

            if rdzv_endpoint is None:
                cmd += f" --rdzv-endpoint={rdzv_endpoint}"

            if max_restarts is not None:
                cmd += f" --max-restarts={max_restarts}"
        else:
            cmd += " --standalone"

        cmd += f" --nnodes={nnodes} --nproc-per-node={nproc_per_node}"
    else:
        cmd = "python"

    print(cmd)


if __name__ == "__main__":
    main()
