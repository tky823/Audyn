audyn.models.ast
================

``audyn.models.ast`` includes Audio Spectrogram Transformer (AST)-related modules.

Classes
-------

Aggregator
^^^^^^^^^^

Module to aggregate features

.. autoclass:: audyn.models.ast.Aggregator
.. autoclass:: audyn.models.ast.AverageAggregator
.. autoclass:: audyn.models.ast.HeadTokensAggregator

Head
^^^^

Module to transform aggregated features

.. autoclass:: audyn.models.ast.Head
.. autoclass:: audyn.models.ast.MLPHead
