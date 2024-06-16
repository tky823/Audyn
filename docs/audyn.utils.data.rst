audyn.utils.data
================

``audyn.utils.data`` provides utilities to process data.

Submodules
----------

.. toctree::
   :maxdepth: 1

   audyn.utils.data.hifigan
   audyn.utils.data.vctk
   audyn.utils.data.birdclef

Create dataset
--------------

.. code-block::

    >>> import os
    >>> import torch
    >>> from audyn.utils.data import TorchObjectDataset
    >>> num_samples = 10
    >>> num_features = 3
    >>> list_path = "./train.txt"
    >>> feature_dir = "./features"
    >>> os.makedirs(feature_dir, exist_ok=True)
    >>> f = open(list_path, mode="w")
    >>> for idx in range(num_samples):
    ...     length = idx + 1
    ...     # NOTE: Last dimension is treated as length.
    ...     data = {
    ...         "index": torch.tensor(idx, dtype=torch.long),
    ...         "value": torch.randn((num_features, length)),
    ...         "filename": f"{idx}",
    ...     }
    ...     path = os.path.join(feature_dir, f"{idx}.pth")
    ...     torch.save(data, path)
    ...     f.write(f"{idx}\n")
    ...
    2
    2
    2
    2
    2
    2
    2
    2
    2
    2
    >>> f.close()
    >>> dataset = TorchObjectDataset(list_path=list_path, feature_dir=feature_dir)
    >>> for sample in dataset:
    ...     print(sample["index"].size(), sample["value"].size(), sample["filename"])
    ... 
    torch.Size([]) torch.Size([3, 1]) 0
    torch.Size([]) torch.Size([3, 2]) 1
    torch.Size([]) torch.Size([3, 3]) 2
    torch.Size([]) torch.Size([3, 4]) 3
    torch.Size([]) torch.Size([3, 5]) 4
    torch.Size([]) torch.Size([3, 6]) 5
    torch.Size([]) torch.Size([3, 7]) 6
    torch.Size([]) torch.Size([3, 8]) 7
    torch.Size([]) torch.Size([3, 9]) 8
    torch.Size([]) torch.Size([3, 10]) 9

Classes
-------

Dataset
^^^^^^^

.. autoclass:: audyn.utils.data.TorchObjectDataset

.. autoclass:: audyn.utils.data.SortableTorchObjectDataset

.. autoclass:: audyn.utils.data.WebDatasetWrapper

Data Loader
^^^^^^^^^^^

.. autoclass:: audyn.utils.data.SequentialBatchDataLoader

.. autoclass:: audyn.utils.data.DynamicBatchDataLoader

.. autoclass:: audyn.utils.data.DistributedDataLoader

.. autoclass:: audyn.utils.data.DistributedSequentialBatchDataLoader

.. autoclass:: audyn.utils.data.DistributedDynamicBatchDataLoader

Composer
^^^^^^^^

.. autoclass:: audyn.utils.data.Composer

.. autoclass:: audyn.utils.data.AudioFeatureExtractionComposer

Collator
^^^^^^^^

.. autoclass:: audyn.utils.data.Collator
