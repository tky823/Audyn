# conftest.py is based on
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
# and
# https://docs.pytest.org/en/latest/deprecations.html#pytest-namespace


from typing import List

import pytest
from dummy.utils import reset_random_port
from pytest import ExitCode, Session


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
    import subprocess
    import warnings

    msg = f"status: {exitstatus}"
    warnings.warn(msg, UserWarning, stacklevel=2)

    process = subprocess.run("ps aux", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    warnings.warn(process.stdout, UserWarning, stacklevel=2)
    warnings.warn(process.stderr, UserWarning, stacklevel=2)
