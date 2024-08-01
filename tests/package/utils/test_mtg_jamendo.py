from audyn.utils.data.mtg_jamendo import (
    genre_metadata,
    genre_tags,
    instrument_metadata,
    instrument_tags,
    moodtheme_metadata,
    moodtheme_tags,
    num_genre_tags,
    num_instrument_tags,
    num_moodtheme_tags,
    num_top50_tags,
    top50_metadata,
    top50_tags,
)


def test_mtg_jamendo() -> None:
    assert len(top50_tags) == num_top50_tags
    assert len(genre_tags) == num_genre_tags
    assert len(instrument_tags) == num_instrument_tags
    assert len(moodtheme_tags) == num_moodtheme_tags

    sample = top50_metadata[30000]

    assert sample["track"] == "track_1098504"
    assert sample["artist"] == "artist_432966"
    assert sample["album"] == "album_131024"
    assert sample["path"] == "04/1098504.mp3"
    assert sample["duration"] == 254.0
    assert sample["tags"] == [
        "genre---chillout",
        "genre---easylistening",
        "genre---electronic",
        "genre---funk",
        "genre---lounge",
        "instrument---bass",
        "instrument---electricpiano",
        "instrument---piano",
        "mood/theme---relaxing",
    ]

    sample = genre_metadata[0]

    assert sample["track"] == "track_0000214"
    assert sample["artist"] == "artist_000014"
    assert sample["album"] == "album_000031"
    assert sample["path"] == "14/214.mp3"
    assert sample["duration"] == 124.6
    assert sample["tags"] == ["genre---punkrock"]

    sample = instrument_metadata[0]

    assert sample["track"] == "track_0000382"
    assert sample["artist"] == "artist_000020"
    assert sample["album"] == "album_000046"
    assert sample["path"] == "82/382.mp3"
    assert sample["duration"] == 211.1
    assert sample["tags"] == ["instrument---voice"]

    sample = moodtheme_metadata[0]

    assert sample["track"] == "track_0000948"
    assert sample["artist"] == "artist_000087"
    assert sample["album"] == "album_000149"
    assert sample["path"] == "48/948.mp3"
    assert sample["duration"] == 212.7
    assert sample["tags"] == ["mood/theme---background"]
