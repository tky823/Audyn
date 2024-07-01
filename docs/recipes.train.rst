Training
========

Template of ``train.py`` for training.

.. code-block:: python

    from omegaconf import DictConfig

    import audyn
    from audyn.utils import setup_config
    from audyn.utils.driver import AutoTrainer


    @audyn.main()
    def main(config: DictConfig) -> None:
        setup_config(config)

        trainer = AutoTrainer.build_from_config(config)
        trainer.run()


    if __name__ == "__main__":
        main()

``AutoTrainer`` tries to instantiate ``config.train.trainer._target_`` class trainer.

Mixed precision training by ``torcf.amp``
-----------------------------------------

`Audyn` supports mixed precision training by ``torcf.amp`` on GPUs.

To activate this feature, please set ``system=cuda_amp``.
If you use mixed precision training with distributed data parallel, please set ``system=cuda_ddp_amp``.

.. code-block:: yaml

    # cuda_amp.yaml
    seed: ...
    distributed:
        ...
    cudnn:
        ...
    amp:  # Please set parameters in this section.
        enable: true
        dtype:  # none by default
    accelerator:

To change ``dtype``, please set ``system.amp.dtype``.
By default, ``system.amp.dtype`` is set to ``None`` and treated as ``torch.float16``.
The following example shows the case where ``dtype`` is ``torch.bfloat16``.

.. code-block:: yaml

    ...
    amp:
        enable: true
        dtype: ${const:torch.bfloat16}
    ...
