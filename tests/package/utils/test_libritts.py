from audyn.utils.data.libritts import num_speakers, speakers


def test_libritts() -> None:
    assert len(speakers) == num_speakers == 2484
