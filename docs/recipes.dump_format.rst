``dump_format``
===============

``dump_format`` parameter in recipes defines the format of data to dump.
Available formats are ``audyn.utils.data.available_dump_formats``.

.. code-block::

   >>> from audyn.utils.data import available_dump_formats
   >>> print(available_dump_formats)
   ['torch', 'webdataset', 'birdclef2024', 'custom']
