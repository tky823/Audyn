Pretrained models via torch.hub
===============================

Audio spectrogram transformer (AST)
-----------------------------------

- The weights are extracted from the original implementation.

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
- The weights are extracted from the original implementation.
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

- The weights are extracted from the original implementation.

.. code-block:: python

    >>> import torch
    >>> repo = "tky823/Audyn"
    >>> model = "passt_base"
    >>> passt = torch.hub.load(
    ...     repo,
    ...     model,
    ...     skip_validation=False,
    ... )

Music Tagging Transformer
-------------------------

- The weights are extracted from the original implementation.
- In terms of reproducibility, it is recommended to load predefined Mel-spectrogram transform as well.
- Teacher model trained by Million Song Dataset (MSD).

.. code-block:: python

    >>> import torch
    >>> repo = "tky823/Audyn"
    >>> model = "teacher_music_tagging_transformer"
    >>> teacher_music_tagging_transformer = torch.hub.load(
    ...     repo,
    ...     model,
    ...     skip_validation=False,
    ... )

.. code-block:: python

    >>> import torch
    >>> repo = "tky823/Audyn"
    >>> transform = "music_tagging_transformer_melspectrogram"
    >>> music_tagging_transformer_melspectrogram = torch.hub.load(
    ...     repo,
    ...     transform,
    ...     skip_validation=False,
    ... )

- Student model trained by MSD.

.. code-block:: python

    >>> import torch
    >>> repo = "tky823/Audyn"
    >>> model = "student_music_tagging_transformer"
    >>> student_music_tagging_transformer = torch.hub.load(
    ...     repo,
    ...     model,
    ...     skip_validation=False,
    ... )
