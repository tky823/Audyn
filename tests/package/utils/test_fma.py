from typing import Any, Dict, Tuple

import pytest
import torch

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
from audyn.utils.data.fma.composer import NAFPWaveformSliceComposer


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


def test_unpacking_composer(fma_samples: Dict[str, Dict[str, Any]]) -> None:
    input_key = "waveform"
    output_key = "waveform_slice"
    shifted_key = "augmented_waveform_slice"

    list_batch = []

    for key, sample in fma_samples.items():
        sample["__key__"] = key
        list_batch.append(sample)

    composer = NAFPWaveformSliceComposer(
        input_key=input_key,
        output_key=output_key,
        shifted_key=shifted_key,
        duration=1,
        offset_duration=0.2,
        sample_rate=16000,
        training=True,
    )
    list_batch = composer(list_batch)

    for sample in list_batch:
        assert {
            "__key__",
            "waveform",
            "waveform_slice",
            "augmented_waveform_slice",
            "sample_rate",
        } == set(sample.keys())


@pytest.fixture
def fma_samples() -> Dict[str, Dict[str, Any]]:
    duration = 5
    sample_rate = 16000

    g = torch.Generator()
    g.manual_seed(0)

    def _create_dummy_audio() -> Tuple[torch.Tensor, int]:
        _duration = torch.randint(int(0.8 * duration), duration, (), generator=g)
        waveform = torch.randn(int(_duration * sample_rate), generator=g)

        return waveform

    samples = {
        "example0": {
            "waveform": _create_dummy_audio(),
            "sample_rate": sample_rate,
        },
        "example1": {
            "waveform": _create_dummy_audio(),
            "sample_rate": sample_rate,
        },
        "example2": {
            "waveform": _create_dummy_audio(),
            "sample_rate": sample_rate,
        },
        "example3": {
            "waveform": _create_dummy_audio(),
            "sample_rate": sample_rate,
        },
        "example4": {
            "waveform": _create_dummy_audio(),
            "sample_rate": sample_rate,
        },
        "example5": {
            "waveform": _create_dummy_audio(),
            "sample_rate": sample_rate,
        },
        "example6": {
            "waveform": _create_dummy_audio(),
            "sample_rate": sample_rate,
        },
        "example7": {
            "waveform": _create_dummy_audio(),
            "sample_rate": sample_rate,
        },
        "example8": {
            "waveform": _create_dummy_audio(),
            "sample_rate": sample_rate,
        },
        "example9": {
            "waveform": _create_dummy_audio(),
            "sample_rate": sample_rate,
        },
        "example10": {
            "waveform": _create_dummy_audio(),
            "sample_rate": sample_rate,
        },
    }

    return samples
