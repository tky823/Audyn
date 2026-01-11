audyn.utils.data.jamendo_max_caps
=================================

``audyn.utils.data.jamendo_max_caps`` provides utilities for JamendoMaxCaps dataset.

.. code-block:: python

    >>> from audyn.utils.data.jamendo_max_caps import download_metadata
    >>> metadata = download_metadata()
    >>> metadata[0].keys()
    dict_keys(['id', 'name', 'duration', 'artist_id', 'artist_name', 'artist_idstr', 'album_name', 'album_id', 'license_ccurl', 'position', 'releasedate', 'album_image', 'audio', 'audiodownload', 'prourl', 'shorturl', 'shareurl', 'waveform', 'image', 'musicinfo', 'audiodownload_allowed', 'captions'])
    >>> metadata[0]["id"]
    '1990389'
    