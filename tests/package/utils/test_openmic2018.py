from audyn.utils.data.openmic2018 import all_tags, download_all_metadata, num_all_tags


def test_openmic2018() -> None:
    all_metadata = download_all_metadata()

    assert len(all_tags) == num_all_tags

    sample = all_metadata[200]

    assert sample["track"] == "001365_157440"
    assert sample["positive"] == ["cymbals", "drums"]
    assert sample["negative"] == ["synthesizer", "trombone", "ukulele"]
    assert sample["unlabeled"] == [
        "accordion",
        "banjo",
        "bass",
        "cello",
        "clarinet",
        "flute",
        "guitar",
        "mallet_percussion",
        "mandolin",
        "organ",
        "piano",
        "saxophone",
        "trumpet",
        "violin",
        "voice",
    ]
    assert sample["subset"] == "test"
