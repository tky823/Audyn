audyn.transforms
================

``audyn.transforms`` provides transformations for audio processing.

Classes
-------

General modules
^^^^^^^^^^^^^^^

.. autoclass:: audyn.transforms.WaveformSlicer
   :members: forward

Librosa-related modules
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.transforms.LibrosaMelSpectrogram
   :members: forward

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

HiFi-GAN-related modules
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.transforms.HiFiGANMelSpectrogram
   :members: forward

AST-related modules
^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.transforms.AudioSpectrogramTransformerMelSpectrogram
   :members: forward

.. autoclass:: audyn.transforms.ASTMelSpectrogram

HuBERT-related modules
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.transforms.HuBERTMFCC
   :members: forward
