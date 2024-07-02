# Audyn
[![codecov](https://codecov.io/gh/tky823/Audyn/graph/badge.svg?token=7R29QDGXLQ)](https://codecov.io/gh/tky823/Audyn)

Audyn is PyTorch toolkit for audio synthesis.

## Build Status

[![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.1.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.1.yaml)

| OS | Python | PyTorch | Status |
|:-:|:-:|:-:|:-:|
| Ubuntu | 3.8 | 2.0 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.8_torch-2.0.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.8_torch-2.0.yaml) |
| Ubuntu | 3.8 | 2.1 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.8_torch-2.1.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.8_torch-2.1.yaml) |
| Ubuntu | 3.8 | 2.2 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.8_torch-2.2.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.8_torch-2.2.yaml) |
| Ubuntu | 3.8 | 2.3 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.8_torch-2.3.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.8_torch-2.3.yaml) |
| Ubuntu | 3.9 | 2.0 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.9_torch-2.0.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.9_torch-2.0.yaml) |
| Ubuntu | 3.9 | 2.1 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.9_torch-2.1.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.9_torch-2.1.yaml) |
| Ubuntu | 3.9 | 2.2 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.9_torch-2.2.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.9_torch-2.2.yaml) |
| Ubuntu | 3.9 | 2.3 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.9_torch-2.3.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.9_torch-2.3.yaml) |
| Ubuntu | 3.10 | 2.0 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.0.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.0.yaml) |
| Ubuntu | 3.10 | 2.1 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.1.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.1.yaml) |
| Ubuntu | 3.10 | 2.2 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.2.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.2.yaml) |
| Ubuntu | 3.10 | 2.3 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.3.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.3.yaml) |
| Ubuntu | 3.11 | 2.0 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.0.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.0.yaml) |
| Ubuntu | 3.11 | 2.1 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.1.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.1.yaml) |
| Ubuntu | 3.11 | 2.2 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.2.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.2.yaml) |
| Ubuntu | 3.11 | 2.3 | [![test package on ubuntu-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.3.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.3.yaml) |
| MacOS | 3.8 | 2.0 | [![test package on macos-13](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.8_torch-2.0.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.8_torch-2.0.yaml) |
| MacOS | 3.8 | 2.1 | [![test package on macos-13](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.8_torch-2.1.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.8_torch-2.1.yaml) |
| MacOS | 3.8 | 2.2 | [![test package on macos-13](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.8_torch-2.2.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.8_torch-2.2.yaml) |
| MacOS | 3.9 | 2.0 | [![test package on macos-13](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.9_torch-2.0.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.9_torch-2.0.yaml) |
| MacOS | 3.9 | 2.1 | [![test package on macos-13](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.9_torch-2.1.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.9_torch-2.1.yaml) |
| MacOS | 3.9 | 2.2 | [![test package on macos-13](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.9_torch-2.2.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.9_torch-2.2.yaml) |
| MacOS | 3.10 | 2.0 | [![test package on macos-13](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.10_torch-2.0.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.10_torch-2.0.yaml) |
| MacOS | 3.10 | 2.1 | [![test package on macos-13](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.10_torch-2.1.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.10_torch-2.1.yaml) |
| MacOS | 3.10 | 2.2 | [![test package on macos-13](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.10_torch-2.2.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.10_torch-2.2.yaml) |
| MacOS | 3.11 | 2.0 | [![test package on macos-13](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.11_torch-2.0.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.11_torch-2.0.yaml) |
| MacOS | 3.11 | 2.1 | [![test package on macos-13](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.11_torch-2.1.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.11_torch-2.1.yaml) |
| Windows | 3.8 | 2.0 | [![test package on windows-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.8_torch-2.0.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.8_torch-2.0.yaml) |
| Windows | 3.8 | 2.1 | [![test package on windows-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.8_torch-2.1.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.8_torch-2.1.yaml) |
| Windows | 3.9 | 2.0 | [![test package on windows-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.9_torch-2.0.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.9_torch-2.0.yaml) |
| Windows | 3.9 | 2.1 | [![test package on windows-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.9_torch-2.1.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.9_torch-2.1.yaml) |
| Windows | 3.10 | 2.0 | [![test package on windows-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.10_torch-2.0.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.10_torch-2.0.yaml) |
| Windows | 3.10 | 2.1 | [![test package on windows-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.10_torch-2.1.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.10_torch-2.1.yaml) |
| Windows | 3.11 | 2.0 | [![test package on windows-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.0.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.0.yaml) |
| Windows | 3.11 | 2.1 | [![test package on windows-latest](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.1.yaml/badge.svg)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.1.yaml) |

## Installation
You can install by pip.
```shell
pip install git+https://github.com/tky823/Audyn.git
```
or clone this repository.
```shell
git clone https://github.com/tky823/Audyn.git
cd Audyn
pip install -e .
```

If you need to run recipes, add `[recipes]` as follows:
```shell
# In Audyn/
pip install -e ".[recipes]"
```

If you use MacOS, you may need to set `MACOSX_DEPLOYMENT_TARGET` during installation to build C++ related modules.

### C++ extension
We use [C++ extension](https://pytorch.org/tutorials/advanced/cpp_extension.html) to search monotonic alignment in some models (e.g. GlowTTS).
To take full advantage of computational efficiency, set appropriate value of `OMP_NUM_THREADS` and `CXX` during installation:

```shell
# In Audyn/
export CXX=<PATH/TO/CPP/COMPILER>  # e.g. /usr/bin/c++
export OMP_NUM_THREADS=<SUITABLE/VALUE/FOR/ENVIRONMENT>
pip install -e "."
```

## Development
```shell
# In Audyn/
pip install -e ".[recipes,dev,tests]"
```

## Build Documentation Locally (optional)
To build the documentation locally, you have to include `docs` when installing `Audyn`.
```shell
pip install -e ".[docs]"
```

When you build the documentation, run the following command.
```shell
cd docs/
make html
```

Or, you can build the documentation automatically using `sphinx-autobuild`.
```shell
# in Audyn/
sphinx-autobuild docs docs/_build/html
```

## Test
```shell
pytest tests/package
```
