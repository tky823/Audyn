audyn.utils.data
================

``audyn.utils.data`` provides utilities to process data.

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
    ...     data = {
    ...         "index": torch.tensor(idx, dtype=torch.long),
    ...         "value": torch.randn((num_features, length)),
    ...         "filename": f"{idx}",
    ...     }
    ...     path = os.path.join(feature_dir, f"{idx}.pth")
    ...     torch.save(data, path)
    ...     f.write(f"{idx}\n")
    ...
    >>> f.close()
    >>> dataset = TorchObjectDataset(list_path=list_path, feature_dir=feature_dir)
    >>> for sample in dataset:
    ...     print(sample["index"].size(), sample["value"].size())
    ... 
    torch.Size([]) torch.Size([3])
    torch.Size([]) torch.Size([3])
    torch.Size([]) torch.Size([3])
    torch.Size([]) torch.Size([3])
    torch.Size([]) torch.Size([3])
    torch.Size([]) torch.Size([3])
    torch.Size([]) torch.Size([3])
    torch.Size([]) torch.Size([3])
    torch.Size([]) torch.Size([3])
    torch.Size([]) torch.Size([3])

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
