# Audyn
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

## Development
```shell
# In Audyn/
pip install -e ".[recipes,dev,tests]"
```

## Test
```shell
pytest tests/audyn
```
