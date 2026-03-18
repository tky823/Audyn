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
from audyn_test.utils import reset_random_port
from pytest import ExitCode, Session

IS_WINDOWS = sys.platform == "win32"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="Run slow tests.")
    parser.addoption("--runddp", action="store_true", default=False, help="Run DDP tests.")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "ddp: mark test as distributed data parallel")

    reset_random_port()


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    run_slow = config.getoption("--runslow")
    run_ddp = config.getoption("--runddp")

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_ddp = pytest.mark.skip(reason="need --runddp option to run")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)

        if "ddp" in item.keywords and not run_ddp:
            item.add_marker(skip_ddp)


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

        error_msg = process.stderr.decode()

        if len(error_msg) > 0:
            logger.error("[STDERR]")
            logger.error(error_msg)

        process = subprocess.run(
            ["pkill", "-u", user, "-f", "torch_shm_manager"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info("[STDOUT]")
        logger.info(process.stdout.decode())

        error_msg = process.stderr.decode()

        if len(error_msg) > 0:
            logger.error("[STDERR]")
            logger.error(error_msg)
