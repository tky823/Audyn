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

Mixed precision training by ``torch.amp``
-----------------------------------------

``Audyn`` supports mixed precision training by ``torch.amp`` on GPUs.

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

Multi-node training
-------------------

``Audyn`` partially supports multi-node training.

To activate this feature, please set ``system=cuda_ddp``.
If you use mixed precision training, please set ``system=cuda_ddp_amp``.

.. code-block:: yaml

    # cuda_amp.yaml
    seed: ...

    distributed:  # Please set parameters in this section.
        enable: true
        nodes:  # number of nodes
        nproc_per_node:  # number of processes per node
        backend: nccl
        init_method: "env://"

        # multi-node training parameters
        rdzv_id:  # required
        rdzv_backend:  # required
        rdzv_endpoint:  # required
        max_restarts:  # optional

    cudnn:
        ...

    amp:
        ...

    accelerator:
