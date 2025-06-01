import os
import shutil
import socket
from urllib.request import Request, urlopen

import pytest

_home_dir = os.path.expanduser("~")
audyn_test_cache_dir = os.getenv("AUDYN_TEST_CACHE_DIR") or os.path.join(
    _home_dir, ".cache", "audyn_test"
)

os.makedirs(audyn_test_cache_dir, exist_ok=True)


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


def clear_cache(**kwargs) -> None:
    """Remove cache directory by ``shutil.rmtree`` for audyn_test.

    Args:
        kwargs: Keyword arguments given to ``shutil.rmtree``.

    """
    shutil.rmtree(audyn_test_cache_dir, **kwargs)
