import os
import uuid
from urllib.request import Request, urlopen

import pytest


def select_random_port() -> int:
    return pytest.random_port


def reset_random_port() -> None:
    max_number = 2**16
    min_number = 1024

    seed = str(uuid.uuid4())
    seed = seed.replace("-", "")
    seed = int(seed, 16)

    pytest.random_port = seed % (max_number - min_number) + min_number

    return pytest.random_port


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
