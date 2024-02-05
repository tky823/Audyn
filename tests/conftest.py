# conftest.py is based on
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
# and
# https://docs.pytest.org/en/latest/deprecations.html#pytest-namespace

import uuid

import pytest


def pytest_configure():
    max_number = 2**16

    seed = str(uuid.uuid4())
    seed = seed.replace("-", "")
    seed = int(seed, 16)

    pytest.random_port = seed % max_number
