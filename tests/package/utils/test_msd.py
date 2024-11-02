from audyn.utils.data.msd_recommend import download_interactions
from audyn.utils.data.msd_tagging import download_metadata, num_tags, tags


def test_msd_tagging() -> None:
    assert len(tags) == num_tags

    metadata = download_metadata(subset="labeled-train")

    assert len(metadata) == 163504


def test_msd_recommend() -> None:
    interactions = download_interactions("user20-track200", subset="validate-visible")

    assert len(interactions) == 50000
    assert interactions["969cc6fb74e076a68e36a04409cb9d3765757508"]["SOABRAB12A6D4F7AAF"] == 2
