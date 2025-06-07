``dump_format``
===============

``dump_format`` parameter in recipes defines the format of data to dump.
Available format list can be displayed by ``audyn.utils.data.list_available_dump_formats``.

.. code-block:: python

   >>> from audyn.utils import list_available_dump_formats
   >>> list_available_dump_formats()
   ['torch', 'webdataset', 'birdclef2024', 'musdb18', 'dnr-v2', 'wordnet', 'fma-small_nafp', 'custom']

- ``custom``: Customized dumping format to avoid errors of format conversion by ``Audyn``.
