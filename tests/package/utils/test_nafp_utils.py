from typing import Any, Dict

import pytest
import torch

from audyn.utils.data.nafp.composer import NAFPWaveformSliceComposer


def test_nafp_composer(fma_samples: Dict[str, Dict[str, Any]]) -> None:
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

        assert sample["waveform_slice"].size() == (16000,)
        assert sample["augmented_waveform_slice"].size() == (16000,)


@pytest.fixture
def fma_samples() -> Dict[str, Dict[str, Any]]:
    duration = 5
    sample_rate = 16000

    g = torch.Generator()
    g.manual_seed(0)

    def _create_dummy_audio() -> torch.Tensor:
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
