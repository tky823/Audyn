# Pretrained models via torch.hub

## Self-supervised audio spectrogram transformer (SSAST)
- Provided weights are extracted from the original implementation.
- SSAST for pretraining

```python
>>> import torch
>>> repo = "tky823/Audyn"
>>> model = "multitask_ssast_base_400"
>>> # patch-based SSAST
>>> token_unit = "patch"
>>> patch_based_ssast = torch.hub.load(
...     repo,
...     model,
...     skip_validation=False,
...     token_unit=token_unit,
... )
>>> # frame-based SSAST
>>> token_unit = "frame"
>>> frame_based_ssast = torch.hub.load(
...     repo,
...     model,
...     skip_validation=False,
...     token_unit=token_unit,
... )
```

- SSAST for finetuning

```python
>>> import torch
>>> repo = "tky823/Audyn"
>>> model = "ssast_base_400"
>>> # patch-based SSAST
>>> token_unit = "patch"
>>> patch_based_ssast = torch.hub.load(
...     repo,
...     model,
...     skip_validation=False,
...     token_unit=token_unit,
... )
>>> # frame-based SSAST
>>> token_unit = "frame"
>>> frame_based_ssast = torch.hub.load(
...     repo,
...     model,
...     skip_validation=False,
...     token_unit=token_unit,
... )
```
