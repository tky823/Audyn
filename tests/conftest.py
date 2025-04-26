# conftest.py is based on
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
# and
# https://docs.pytest.org/en/latest/deprecations.html#pytest-namespace


import getpass
import logging
import subprocess
import sys
from typing import List

import pytest
from dummy.utils import reset_random_port
from pytest import ExitCode, Session

IS_WINDOWS = sys.platform == "win32"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="Run slow tests.")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")

    reset_random_port()


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="Need --runslow option to run")

    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_sessionfinish(session: Session, exitstatus: ExitCode) -> None:
    """Kill torch_shm_manager process for Ubuntu & MacOS in GHA.

    See https://github.com/tky823/Audyn/pull/271.
    """
    logger = logging.getLogger(__name__)

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    logger.info(f"status: {exitstatus}")

    if not IS_WINDOWS:
        user = getpass.getuser()
        process = subprocess.run(["ps", "aux"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        logger.info("[STDOUT]")
        logger.info(process.stdout.decode())

        logger.info("[STDERR]")
        logger.info(process.stderr.decode())

        process = subprocess.run(
            ["pkill", "-u", user, "-f", "torch_shm_manager"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info("[STDOUT]")
        logger.info(process.stdout.decode())

        logger.info("[STDERR]")
        logger.info(process.stderr.decode())
