audyn.transforms
================

``audyn.transforms`` provides transformations for audio processing.

Classes
-------

Kaldi-related transforms
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.transforms.KaldiMelSpectrogram
.. autoclass:: audyn.transforms.KaldiMFCC

Constant-Q transform
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.transforms.ConstantQTransform
.. autoclass:: audyn.transforms.CQT

AST-related modules
^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.transforms.SelfSupervisedAudioSpectrogramTransformerMelSpectrogram
.. autoclass:: audyn.transforms.SSASTMelSpectrogram

HuBERT-related modules
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.transforms.HuBERTMFCC
