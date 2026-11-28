audyn.models.bsroformer
=======================

``audyn.models.bsroformer`` includes Band-split RoFormer (BSRoFormer)-related models.
The models take a complex spectrogram and return a separated spectrogram with the
same frequency and time dimensions. For multi-source separation, the output has
an additional source dimension.

Pretrained Mel-BSRoFormer
-------------------------

``AudioSeparatorCompatibleBandSplitRoFormer`` can load the pretrained
Mel-BSRoFormer compatible with `audio-separator
<https://github.com/nomadkaraoke/python-audio-separator>`_. The input is a complex
spectrogram with 1025 frequency bins and one channel.

.. code-block:: python

    >>> import torch
    >>> from audyn.models import AudioSeparatorCompatibleBandSplitRoFormer
    >>> model = AudioSeparatorCompatibleBandSplitRoFormer.build_from_pretrained(
    ...     "audio-separator_melbsroformer"
    ... )
    >>> model.eval()
    >>> spectrogram = torch.randn(1, 1, 1025, 8, dtype=torch.complex64)
    >>> with torch.no_grad():
    ...     separated_spectrogram = model(spectrogram)
    >>> print(separated_spectrogram.shape)
    torch.Size([1, 1, 1025, 8])

The first call downloads the checkpoint. A local checkpoint path is also accepted
by ``build_from_pretrained``.
