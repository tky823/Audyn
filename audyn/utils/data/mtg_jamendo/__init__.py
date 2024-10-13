from ._download import (
    download_all_metadata,
    download_all_tags,
    download_genre_metadata,
    download_genre_tags,
    download_instrument_metadata,
    download_instrument_tags,
    download_moodtheme_metadata,
    download_moodtheme_tags,
    download_top50_metadata,
    download_top50_tags,
)
from .collater import MTGJamendoCollator
from .composer import MTGJamendoEvaluationWaveformSliceComposer

__all__ = [
    "all_tags",
    "top50_tags",
    "genre_tags",
    "instrument_tags",
    "moodtheme_tags",
    "num_all_tags",
    "num_top50_tags",
    "num_genre_tags",
    "num_instrument_tags",
    "num_moodtheme_tags",
    "download_all_metadata",
    "download_top50_metadata",
    "download_genre_metadata",
    "download_instrument_metadata",
    "download_moodtheme_metadata",
    "MTGJamendoEvaluationWaveformSliceComposer",
    "MTGJamendoCollator",
]

all_tags = download_all_tags()
top50_tags = download_top50_tags()
genre_tags = download_genre_tags()
instrument_tags = download_instrument_tags()
moodtheme_tags = download_moodtheme_tags()
num_all_tags = len(all_tags)
num_top50_tags = len(top50_tags)
num_genre_tags = len(genre_tags)
num_instrument_tags = len(instrument_tags)
num_moodtheme_tags = len(moodtheme_tags)
