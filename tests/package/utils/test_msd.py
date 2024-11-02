from audyn.utils.data.msd_tagging import download_interactions, num_tags, tags


def test_msd() -> None:
    assert len(tags) == num_tags

    interactions = download_interactions("user20-track200", subset="validate-visible")

    assert len(interactions) == 50000
    assert interactions["969cc6fb74e076a68e36a04409cb9d3765757508"]["SOABRAB12A6D4F7AAF"] == 2
