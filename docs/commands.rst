Commands
========

Decode MUSDB18 .mp4 files
-------------------------

You can decode .stem.mp4 file(s) into .wav files for MUSDB18 dataset by ``audyn-decode-musdb18``.

.. autofunction:: audyn.bin.decode_musdb18.main


Download MTG-Jamendo dataset
----------------------------

You can download audio files of MTG-Jamendo dataset by ``audyn-download-mtg-jamando``.

.. code-block:: shell

    server_type="mirror"  # or "origin"
    quality="raw"  # or "low"
    root="./MTG-Jamendo/raw"  # root directory to store
    chunk_size=1024  # chunk size in byte to download

    audyn-download-mtg-jamando \
    server_type="${server_type}" \
    quality="${quality}" \
    root="${root}" \
    chunk_size=${chunk_size}

Then, please unpack tar files under ``./MTG-Jamendo/raw``.

Download OpenMIC-2018
---------------------

You can download OpenMIC-2018 dataset by ``audyn-download-openmic2018``.

.. code-block:: shell

    data_root="./"  # root directory to save .zip file.
    openmic2018_root="${data_root}/openmic-2018"
    unpack=true  # unpack .tgz or not
    chunk_size=8192  # chunk size in byte to download

    audyn-download-openmic2018 \
    root="${data_root}" \
    openmic2018_root="${openmic2018_root}" \
    unpack=${unpack} \
    chunk_size=${chunk_size}

Then, please unpack tar files under ``./openmic-2018``.
