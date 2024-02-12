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

## Development
```shell
# In Audyn/
pip install -e ".[recipes,dev,tests]"
```

## Test
```shell
pytest tests/package
```
