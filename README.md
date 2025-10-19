# Audyn
[![codecov](https://codecov.io/gh/tky823/Audyn/graph/badge.svg?token=7R29QDGXLQ)](https://codecov.io/gh/tky823/Audyn)

Audyn is PyTorch toolkit for audio synthesis.

## Build Status

| Python/PyTorch | Ubuntu | MacOS (x86_64) | MacOS (arm64) | Windows |
|:-:|:-:|:-:|:-:|:-:|
| 3.10/2.0 | [![ubuntu-latest/3.10/2.0](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.0.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.0.yaml) | [![macos-13/3.10/2.0](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.10_torch-2.0.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.10_torch-2.0.yaml) |  | [![windows-latest/3.10/2.0](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.10_torch-2.0.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.10_torch-2.0.yaml) |
| 3.10/2.1 | [![ubuntu-latest/3.10/2.1](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.1.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.1.yaml) | [![macos-13/3.10/2.1](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.10_torch-2.1.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.10_torch-2.1.yaml) |  | [![windows-latest/3.10/2.1](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.10_torch-2.1.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.10_torch-2.1.yaml) |
| 3.10/2.2 | [![ubuntu-latest/3.10/2.2](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.2.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.2.yaml) | [![macos-13/3.10/2.2](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.10_torch-2.2.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.10_torch-2.2.yaml) |  | [![windows-latest/3.10/2.2](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.10_torch-2.2.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.10_torch-2.2.yaml) |
| 3.10/2.3 | [![ubuntu-latest/3.10/2.3](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.3.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.10_torch-2.3.yaml) |  |  |  |
| 3.11/2.0 | [![ubuntu-latest/3.11/2.0](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.0.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.0.yaml) | [![macos-13/3.11/2.0](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.11_torch-2.0.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.11_torch-2.0.yaml) |  | [![windows-latest/3.11/2.0](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.0.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.0.yaml) |
| 3.11/2.1 | [![ubuntu-latest/3.11/2.1](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.1.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.1.yaml) | [![macos-13/3.11/2.1](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.11_torch-2.1.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.11_torch-2.1.yaml) |  | [![windows-latest/3.11/2.1](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.1.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.1.yaml) |
| 3.11/2.2 | [![ubuntu-latest/3.11/2.2](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.2.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.2.yaml) | [![macos-13/3.11/2.2](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.11_torch-2.2.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.11_torch-2.2.yaml) |  | [![windows-latest/3.11/2.2](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.2.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.2.yaml) |
| 3.11/2.3 | [![ubuntu-latest/3.11/2.3](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.3.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.3.yaml) |  | [![macos-latest/3.11/2.3](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.11_torch-2.3.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.11_torch-2.3.yaml)  |  |
| 3.11/2.4 | [![ubuntu-latest/3.11/2.4](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.4.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.4.yaml) |  | [![macos-latest/3.11/2.4](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.11_torch-2.4.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.11_torch-2.4.yaml) |  |
| 3.11/2.5 | [![ubuntu-latest/3.11/2.5](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.5.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.5.yaml) |  | [![macos-latest/3.11/2.5](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.11_torch-2.5.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.11_torch-2.5.yaml) | [![windows-latest/3.11/2.5](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.5.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.5.yaml) |
| 3.11/2.6 | [![ubuntu-latest/3.11/2.6](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.6.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.11_torch-2.6.yaml) |  | [![macos-latest/3.11/2.6](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.11_torch-2.6.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.11_torch-2.6.yaml) | [![windows-latest/3.11/2.6](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.6.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.11_torch-2.6.yaml) |
| 3.12/2.2 | [![ubuntu-latest/3.12/2.2](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.12_torch-2.2.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.12_torch-2.2.yaml) | [![macos-13/3.12/2.2](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.12_torch-2.2.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.12_torch-2.2.yaml) |  | [![windows-latest/3.12/2.2](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.12_torch-2.2.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.12_torch-2.2.yaml) |
| 3.12/2.3 | [![ubuntu-latest/3.12/2.3](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.12_torch-2.3.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.12_torch-2.3.yaml) |  | [![macos-latest/3.12/2.3](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.12_torch-2.3.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.12_torch-2.3.yaml) |  |
| 3.12/2.4 | [![ubuntu-latest/3.12/2.4](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.12_torch-2.4.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.12_torch-2.4.yaml) |  | [![macos-latest/3.12/2.4](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.12_torch-2.4.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.12_torch-2.4.yaml) |  |
| 3.12/2.5 | [![ubuntu-latest/3.12/2.5](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.12_torch-2.5.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.12_torch-2.5.yaml) |  | [![macos-latest/3.12/2.5](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.12_torch-2.5.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.12_torch-2.5.yaml) | [![windows-latest/3.12/2.5](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.12_torch-2.5.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.12_torch-2.5.yaml) |
| 3.12/2.6 | [![ubuntu-latest/3.12/2.6](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.12_torch-2.6.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.12_torch-2.6.yaml) |  | [![macos-latest/3.12/2.6](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.12_torch-2.6.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.12_torch-2.6.yaml) | [![windows-latest/3.12/2.6](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.12_torch-2.6.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.12_torch-2.6.yaml) |
| 3.13/2.2 | [![ubuntu-latest/3.13/2.2](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.13_torch-2.2.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.13_torch-2.2.yaml) | [![macos-13/3.13/2.2](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.13_torch-2.2.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-13_python-3.13_torch-2.2.yaml) | [![macos-latest/3.13/2.2](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.13_torch-2.2.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.13_torch-2.2.yaml) | [![windows-latest/3.13/2.2](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.13_torch-2.2.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.13_torch-2.2.yaml) |
| 3.13/2.6 | [![ubuntu-latest/3.13/2.6](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.13_torch-2.6.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.13_torch-2.6.yaml) |  | [![macos-latest/3.13/2.6](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.13_torch-2.6.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.13_torch-2.6.yaml) | [![windows-latest/3.13/2.6](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.13_torch-2.6.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.13_torch-2.6.yaml) |
| 3.13/2.7 | [![ubuntu-latest/3.13/2.7](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.13_torch-2.7.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.13_torch-2.7.yaml) |  | [![macos-latest/3.13/2.7](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.13_torch-2.7.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.13_torch-2.7.yaml) | [![windows-latest/3.13/2.7](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.13_torch-2.7.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_windows-latest_python-3.13_torch-2.7.yaml) |
| 3.13/2.8 | [![ubuntu-latest/3.13/2.8](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.13_torch-2.8.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_ubuntu-latest_python-3.13_torch-2.8.yaml) |  | [![macos-latest/3.13/2.8](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.13_torch-2.8.yaml/badge.svg?branch=main)](https://github.com/tky823/Audyn/actions/workflows/test_package_macos-latest_python-3.13_torch-2.8.yaml) |  |

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
git clone https://github.com/tky823/Audyn.git
cd Audyn
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

To include slow tests
```shell
pytest tests/package --runslow
```

## License
- Apache License, Version 2.0 **EXCEPT FOR WEIGHTS OF PRETRAINED MODELS**
- Weights for some of the pre-trained models are extracted from the official implementations. Their licenses follow the official implementations.
