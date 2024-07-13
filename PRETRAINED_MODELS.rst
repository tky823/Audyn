Pretrained models via torch.hub
===============================

Audio spectrogram transformer (AST)
-----------------------------------

- Provided weights are extracted from the original implementation.

.. code-block:: python
   
    >>> import torch
    >>> repo = "tky823/Audyn"
    >>> model = "ast_base"
    >>> ast = torch.hub.load(
    >>>     repo,
    >>>     model,
    >>>     skip_validation=False,
    >>> )

Self-supervised audio spectrogram transformer (SSAST)
-----------------------------------------------------
- Provided weights are extracted from the original implementation.
- SSAST for pretraining

.. code-block:: python

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

- SSAST for finetuning

.. code-block:: python

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

Patchout faSt Spectrogram Transformer (PaSST)
---------------------------------------------

.. code-block:: python

    >>> import torch
    >>> repo = "tky823/Audyn"
    >>> model = "passt_base"
    >>> passt = torch.hub.load(
    ...     repo,
    ...     model,
    ...     skip_validation=False,
    ... )
