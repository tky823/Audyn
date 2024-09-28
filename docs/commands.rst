Commands
========

Download MUSDB18 dataset
----------------------------

You can download MUSDB18 dataset by ``audyn-download-musdb18``.

.. code-block:: shell

    type="default"  # for MUSDB18
    # type="hq"  # for MUSDB18-HQ
    # type="7s"  # for MUSDB18-7s

    data_root="./data"  # root directory to save .zip file.
    musdb18_root="${data_root}/MUSDB18"
    unpack=true  # unpack .zip or not
    chunk_size=8192  # chunk size in byte to download

    audyn-download-musdb18 \
    type="${type}" \
    root="${data_root}" \
    musdb18_root="${musdb18_root}" \
    unpack=${unpack} \
    chunk_size=${chunk_size}

Decode MUSDB18 .mp4 files
-------------------------

You can decode .stem.mp4 file(s) into .wav files for MUSDB18 dataset by ``audyn-decode-musdb18``.

.. autofunction:: audyn.bin.decode_musdb18.main


Download MTG-Jamendo dataset
----------------------------

You can download audio files of MTG-Jamendo dataset by ``audyn-download-mtg-jamendo``.

.. code-block:: shell

    server_type="mirror"  # or "origin"
    quality="raw"  # or "low"
    root="./MTG-Jamendo/raw"  # root directory to store
    unpack=true  # unpack .tar or not
    chunk_size=1024  # chunk size in byte to download

    audyn-download-mtg-jamendo \
    server_type="${server_type}" \
    quality="${quality}" \
    root="${root}" \
    unpack=${unpack} \
    chunk_size=${chunk_size}

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
