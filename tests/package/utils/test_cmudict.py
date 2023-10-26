from audyn.utils.data.cmudict import CMUDict


def test_cmudict() -> None:
    pron_dict = CMUDict()

    assert pron_dict["Hello"] == ["HH", "AH0", "L", "OW1"]
