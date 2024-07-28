audyn.utils.data.mtg_jamendo
============================

``audyn.utils.data.mtg_jamendo`` provides utilities for [MTG-Jamendo dataset](https://mtg.github.io/mtg-jamendo-dataset/).

.. code-block:: python

    >>> from audyn.utils.data.mtg_jamendo import top50_tags, num_top50_tags
    >>> top50_tags[-7:-3]
    ['instrument---violin', 'instrument---voice', 'mood/theme---emotional', 'mood/theme---energetic']
    >>> print(num_top50_tags)
    50
    >>> from audyn.utils.data.mtg_jamendo import genre_tags, instrument_tags, moodtheme_tags
    >>> print(genre_tags[:5])
    ['genre---60s', 'genre---70s', 'genre---80s', 'genre---90s', 'genre---acidjazz']
    >>> print(instrument_tags[:5])
    ['instrument---accordion', 'instrument---acousticbassguitar', 'instrument---acousticguitar', 'instrument---bass', 'instrument---beat']
    >>> print(moodtheme_tags[:5])
    ['mood/theme---action', 'mood/theme---adventure', 'mood/theme---advertising', 'mood/theme---ambiental', 'mood/theme---background']
