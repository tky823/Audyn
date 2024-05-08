# Audyn
[![codecov](https://codecov.io/gh/tky823/Audyn/graph/badge.svg?token=7R29QDGXLQ)](https://codecov.io/gh/tky823/Audyn)

Audyn is PyTorch toolkit for audio synthesis.

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
