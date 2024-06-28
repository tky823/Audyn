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

Conv-TasNet
^^^^^^^^^^^

.. autoclass:: audyn.models.ConvTasNet
   :members: forward

WaveNet
^^^^^^^

.. autoclass:: audyn.models.WaveNet
   :members: forward
   
.. autoclass:: audyn.models.MultiSpeakerWaveNet
   :members: forward

WaveGlow
^^^^^^^^

.. autoclass:: audyn.models.WaveGlow
   :members: forward

.. autoclass:: audyn.models.MultiSpeakerWaveGlow
   :members: forward

Audio spectrogram Transformer (AST) and self-supervised AST (SSAST)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.models.AudioSpectrogramTransformer
   :members: forward

.. autoclass:: audyn.models.AST

.. autoclass:: audyn.models.MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel
   :members: forward

.. autoclass:: audyn.models.SelfSupervisedAudioSpectrogramTransformer
   :members: forward

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
