audyn.utils
===========

``audyn.utils`` provides utilities for ``Audyn``.

Submodules
----------

.. toctree::
   :maxdepth: 1

   audyn.utils.driver
   audyn.utils.data
   audyn.utils.modules
   audyn.utils.music

Cache directory
---------------

By default, ``$HOME/.cache/audyn`` is used as a cache directory of ``Audyn``.
To change the cache directory, set ``AUDYN_CACHE_DIR`` as an environmental variable **before** importing ``Audyn``.
To remove contents saved in ``AUDYN_CACHE_DIR``, call ``audyn.utils.clear_cache()``.

Functions
---------

.. autofunction:: audyn.utils.clear_cache

.. autofunction:: audyn.utils.setup_system

.. autofunction:: audyn.utils.setup_config

.. autofunction:: audyn.utils.set_seed

.. autofunction:: audyn.utils.convert_dataset_and_dataloader_to_ddp_if_possible

.. autofunction:: audyn.utils.convert_dataset_and_dataloader_format_if_necessary
