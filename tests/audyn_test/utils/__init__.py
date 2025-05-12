import os
import socket
from urllib.request import Request, urlopen

import pytest


def select_random_port() -> int:
    return pytest.random_port


def reset_random_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    _, port = sock.getsockname()

    pytest.random_port = port

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
