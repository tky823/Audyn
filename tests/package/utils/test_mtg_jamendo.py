from audyn.utils.data.mtg_jamendo import (
    genre_tags,
    instrument_tags,
    moodtheme_tags,
    num_genre_tags,
    num_instrument_tags,
    num_moodtheme_tags,
    num_top50_tags,
    top50_tags,
)


def test_mtg_jamendo() -> None:
    assert len(top50_tags) == num_top50_tags
    assert len(genre_tags) == num_genre_tags
    assert len(instrument_tags) == num_instrument_tags
    assert len(moodtheme_tags) == num_moodtheme_tags
