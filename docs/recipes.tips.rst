Tips of recipes
===============

Distributed training with fixed total batch size
------------------------------------------------

To train model with DDP, the value of ``batch_size`` is shared among devices.
If you want to change the number of devices while keeping the value of ``batch_size`` fixed, configure as follows:

.. code-block:: yaml

    batch_size:
      _target_: operator.floordiv
      _args_:
        - 64
        - _target_: builtins.int
        - _args_: ${oc.env:WORLD_SIZE}

In this configuration, the batch size per GPU is ``64 // WORLD_SIZE``.

Set constant value to config
----------------------------

To set constant (e.g. number of classes in AudioSet) to hydra config, many people tend to write its value directory.

.. code-block:: yaml

    # audyn.utils.data.audioset.num_tags also returns 527.
    num_classes: 527

``Audyn`` supports a customized resolver to use constants by setting ``const:`` prefix:

.. code-block:: yaml

    # Audyn supports following syntax.
    num_classes: ${const:audyn.utils.data.audioset.num_tags}
