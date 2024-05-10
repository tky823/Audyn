Tips of recipes
===============

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
