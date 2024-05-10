audyn.models
============

``audyn.models`` provides several models for audio processing.

Submodules
----------

.. toctree::
   :maxdepth: 1

   audyn.models.wavenet
   audyn.models.waveglow
   audyn.models.hifigan
   audyn.models.ast
   audyn.models.ssast
   audyn.models.passt
   audyn.models.roformer

Classes
-------

WaveNet
^^^^^^^

.. autoclass:: audyn.models.WaveNet
.. autoclass:: audyn.models.MultiSpeakerWaveNet

WaveGlow
^^^^^^^^

.. autoclass:: audyn.models.WaveGlow
.. autoclass:: audyn.models.MultiSpeakerWaveGlow

Audio spectrogram Transformer (AST) and self-supervised AST (SSAST)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.models.AudioSpectrogramTransformer
.. autoclass:: audyn.models.AST
.. autoclass:: audyn.models.MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel
.. autoclass:: audyn.models.SelfSupervisedAudioSpectrogramTransformer
.. autoclass:: audyn.models.SSASTMPM
.. autoclass:: audyn.models.SSAST

Patchout faSt Spectrogram Transformer (PaSST)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.models.PaSST

Rotary Transformer (RoFormer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: audyn.models.RoFormerEncoderLayer
.. autoclass:: audyn.models.RoFormerDecoderLayer
.. autoclass:: audyn.models.RoFormerEncoder
.. autoclass:: audyn.models.RoFormerDecoder
