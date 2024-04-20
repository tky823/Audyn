import os
from urllib.request import Request, urlopen

import pytest
import torch


def select_random_port() -> int:
    return pytest.random_port


def set_ddp_environment(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    num_threads = torch.get_num_threads()
    num_threads = max(num_threads // world_size, 1)
    torch.set_num_threads(num_threads)


def download_file(url: str, save_dir: str, chunk_size: int = 8192) -> str:
    filename = os.path.basename(url)
    path = os.path.join(save_dir, filename)

    os.makedirs(save_dir, exist_ok=True)

    request = Request(url)

    with urlopen(request) as response, open(path, "wb") as f:
        while True:
            chunk = response.read(chunk_size)

            if not chunk:
                break

            f.write(chunk)

    return path
