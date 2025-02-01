from audyn.utils.data.openmic2018 import all_tags, num_all_tags


def test_openmic2018() -> None:
    assert len(all_tags) == num_all_tags
