audyn.models.ast
================

``audyn.models.ast`` includes Audio Spectrogram Transformer (AST)-related modules.

Classes
-------

AST
^^^

Base class of AST

.. autoclass:: audyn.models.ast.BaseAudioSpectrogramTransformer
   :members: compute_padding_mask, patch_transformer_forward, transformer_forward
      spectrogram_to_patches, patches_to_sequence, sequence_to_patches, split_sequence, prepend_tokens

Aggregator
^^^^^^^^^^

Module to aggregate features

.. autoclass:: audyn.models.ast.Aggregator
   :members: forward

.. autoclass:: audyn.models.ast.AverageAggregator
   :members: forward

.. autoclass:: audyn.models.ast.HeadTokensAggregator
   :members: forward

Head
^^^^

Module to transform aggregated features

.. autoclass:: audyn.models.ast.Head
   :members: forward

.. autoclass:: audyn.models.ast.MLPHead
   :members: forward
