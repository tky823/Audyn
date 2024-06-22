from audyn.utils.data.vctk import num_speakers, num_valid_speakers, speakers, valid_speakers


def test_vctk() -> None:
    assert len(speakers) == num_speakers == 110
    assert len(valid_speakers) == num_valid_speakers == 108
