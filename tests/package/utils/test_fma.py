from audyn.utils.data.fma import (
    full_test_track_ids,
    full_train_track_ids,
    full_validation_track_ids,
    large_test_track_ids,
    large_train_track_ids,
    large_validation_track_ids,
    medium_test_track_ids,
    medium_train_track_ids,
    medium_validation_track_ids,
    small_test_track_ids,
    small_train_track_ids,
    small_validation_track_ids,
)


def test_fma() -> None:
    num_small_train_track_ids = len(small_train_track_ids)
    num_small_validation_track_ids = len(small_validation_track_ids)
    num_small_test_track_ids = len(small_test_track_ids)

    assert num_small_train_track_ids == 6400
    assert num_small_validation_track_ids == 800
    assert num_small_test_track_ids == 800

    num_medium_train_track_ids = len(medium_train_track_ids)
    num_medium_validation_track_ids = len(medium_validation_track_ids)
    num_medium_test_track_ids = len(medium_test_track_ids)

    assert num_medium_train_track_ids == 13522
    assert num_medium_validation_track_ids == 1705
    assert num_medium_test_track_ids == 1773

    num_large_train_track_ids = len(large_train_track_ids)
    num_large_validation_track_ids = len(large_validation_track_ids)
    num_large_test_track_ids = len(large_test_track_ids)

    assert num_large_train_track_ids == 64431
    assert num_large_validation_track_ids == 8453
    assert num_large_test_track_ids == 8690

    num_full_train_track_ids = len(full_train_track_ids)
    num_full_validation_track_ids = len(full_validation_track_ids)
    num_full_test_track_ids = len(full_test_track_ids)

    assert (
        num_full_train_track_ids
        == num_small_train_track_ids + num_medium_train_track_ids + num_large_train_track_ids
    )
    assert (
        num_full_validation_track_ids
        == num_small_validation_track_ids
        + num_medium_validation_track_ids
        + num_large_validation_track_ids
    )
    assert (
        num_full_test_track_ids
        == num_small_test_track_ids + num_medium_test_track_ids + num_large_test_track_ids
    )
