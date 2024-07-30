Commands
========

Download MTG-Jamendo dataset
----------------------------

You can download audio files of MTG-Jamendo dataset by ``audyn-download-mtg-jamando``.

.. code-block:: shell

    server_type="mirror"  # or "origin"
    quality="raw"  # or "low"
    output="./MTG-Jamendo/raw"  # output directory to store
    chunk_size=1024  # chunk size in byte to download

    audyn-download-mtg-jamando \
    server_type="${server_type}" \
    quality="${quality}" \
    output="${output}" \
    chunk_size=${chunk_size}

Then, please unpack tar files under ``./MTG-Jamendo/raw``.
