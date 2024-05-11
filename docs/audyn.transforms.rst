audyn.transforms
================

``audyn.transforms`` provides transformations for audio processing.

Classes
-------

Kaldi-related transforms
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.transforms.KaldiMelSpectrogram
   :members: forward

.. autoclass:: audyn.transforms.KaldiMFCC
   :members: forward

Constant-Q transform
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.transforms.ConstantQTransform
   :members: forward

.. autoclass:: audyn.transforms.CQT

AST-related modules
^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.transforms.SelfSupervisedAudioSpectrogramTransformerMelSpectrogram
   :members: forward

.. autoclass:: audyn.transforms.SSASTMelSpectrogram

HuBERT-related modules
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.transforms.HuBERTMFCC
   :members: forward
