import pytest

from audyn.utils.data.jamendo_max_caps import download_metadata


@pytest.mark.slow
def test_download_jamendo_max_caps_metadata() -> None:
    metadata = download_metadata()

    assert len(metadata) == 394198
