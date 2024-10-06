from audyn.utils.data.mtg_jamendo import (
    download_genre_metadata,
    download_instrument_metadata,
    download_moodtheme_metadata,
    download_top50_metadata,
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
    top50_metadata = download_top50_metadata()
    genre_metadata = download_genre_metadata()
    instrument_metadata = download_instrument_metadata()
    moodtheme_metadata = download_moodtheme_metadata()

    assert len(top50_tags) == num_top50_tags
    assert len(genre_tags) == num_genre_tags
    assert len(instrument_tags) == num_instrument_tags
    assert len(moodtheme_tags) == num_moodtheme_tags

    sample = top50_metadata[15000]

    assert sample["track"] == "track_1031094"
    assert sample["artist"] == "artist_433491"
    assert sample["album"] == "album_121295"
    assert sample["path"] == "94/1031094.mp3"
    assert sample["duration"] == 198.6
    assert sample["tags"] == [
        "genre---electronic",
        "genre---experimental",
        "instrument---piano",
        "instrument---synthesizer",
    ]

    sample = genre_metadata[0]

    assert sample["track"] == "track_0000241"
    assert sample["artist"] == "artist_000005"
    assert sample["album"] == "album_000033"
    assert sample["path"] == "41/241.mp3"
    assert sample["duration"] == 340.1
    assert sample["tags"] == ["genre---rock"]

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
