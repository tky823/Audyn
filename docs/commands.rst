Commands
========

Download LJSpeech dataset
-------------------------

You can download LJSpeech dataset by ``audyn-download-ljspeech``.

.. code-block:: shell

    data_root="./data"  # root directory to save .zip file.
    ljspeech_root="${data_root}/LJSpeech-1.1"
    unpack=true  # unpack .tar.bz2 or not
    chunk_size=8192  # chunk size in byte to download

    audyn-download-ljspeech \
    root="${data_root}" \
    ljspeech_root="${ljspeech_root}" \
    unpack=${unpack} \
    chunk_size=${chunk_size}

Download MUSDB18 dataset
------------------------

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


Download FSD50K dataset
-----------------------

You can download FSD50K dataset by ``audyn-download-fsd50k``.

.. code-block:: shell

    data_root="./data"  # root directory to save .zip file.
    fsd50k_root="${data_root}/FSD50K"
    unpack=true  # unpack .zip or not
    chunk_size=8192  # chunk size in byte to download

    audyn-download-fsd50k \
    root="${data_root}" \
    fsd50k_root="${fsd50k_root}" \
    unpack=${unpack} \
    chunk_size=${chunk_size}


Download LSX dataset
--------------------

You can download LSX dataset by ``audyn-download-lsx``.

.. code-block:: shell

    data_root="./data"  # root directory to save .zip file.
    lsx_root="${data_root}/lsx"
    unpack=true  # unpack .zip or not
    chunk_size=8192  # chunk size in byte to download

    audyn-download-lsx \
    root="${data_root}" \
    lsx_root="${lsx_root}" \
    unpack=${unpack} \
    chunk_size=${chunk_size}


Download DnR dataset
--------------------

You can download DnR dataset dataset by ``audyn-download-dnr``.
Only version 2 is supported.

.. code-block:: shell

    data_root="./data"  # root directory to save .tar.gz file.
    dnr_root="${data_root}/DnR-V2"
    version=2
    unpack=true  # unpack .tar.gz or not
    chunk_size=8192  # chunk size in byte to download

    audyn-download-dnr \
    root="${data_root}" \
    dnr_root="${dnr_root}" \
    version=${version} \
    unpack=${unpack} \
    chunk_size=${chunk_size}


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


Download MagnaTagATune (MTAT) dataset
-------------------------------------

You can download audio files of MTAT dataset by ``audyn-download-mtat``.

.. code-block:: shell

    data_root="./data"  # root directory to save .zip file.
    mtat_root="${data_root}/MTAT"
    unpack=true  # unpack .zip or not
    chunk_size=8192  # chunk size in byte to download

    audyn-download-mtat \
    root="${data_root}" \
    mtat_root="${mtat_root}" \
    unpack=${unpack} \
    chunk_size=${chunk_size}


Download OpenMIC-2018 dataset
-----------------------------

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


Download FMA dataset
--------------------

You can download FMA dataset by ``audyn-download-fma``.

.. code-block:: shell

    type="medium"  # for FMA-medium

    data_root="./data"  # root directory to save .zip file.
    fma_root="${data_root}/FMA/${type}"
    unpack=true  # unpack .zip or not
    chunk_size=8192  # chunk size in byte to download

    audyn-download-fma \
    type="${type}" \
    root="${data_root}" \
    fma_root="${fma_root}" \
    unpack=${unpack} \
    chunk_size=${chunk_size}


Download SingMOS dataset
------------------------

You can download SingMOS dataset by ``audyn-download-singmos``.

.. code-block:: shell

    data_root="./data"
    singmos_root="${data_root}/SingMOS"
    chunk_size=8192  # chunk size in byte to download

    audyn-download-singmos \
    singmos_root="${singmos_root}" \
    chunk_size=${chunk_size}


Download Song Describer dataset
-------------------------------

You can download Song Describer dataset by ``audyn-download-song-describer``.

.. code-block:: shell

    data_root="./data"
    song_describer_root="${data_root}/SongDescriber"
    chunk_size=8192  # chunk size in byte to download

    audyn-download-song-describer \
    song_describer_root="${song_describer_root}" \
    chunk_size=${chunk_size}
