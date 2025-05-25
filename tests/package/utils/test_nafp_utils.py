from typing import Any, Dict

import pytest
import torch
import torchaudio.transforms as aT

from audyn.utils.data import AudioFeatureExtractionComposer
from audyn.utils.data.nafp.composer import (
    NAFPSpectrogramAugmentationComposer,
    NAFPWaveformSliceComposer,
)


def test_nafp_composer(fma_samples: Dict[str, Dict[str, Any]]) -> None:
    input_key = "waveform"
    output_key = "waveform_slice"
    shifted_key = "augmented_waveform_slice"
    melspectrogram_key = "melspectrogram_slice"
    sample_rate = 16000
    duration = 1

    list_batch = []

    for key, sample in fma_samples.items():
        sample["__key__"] = key
        list_batch.append(sample)

    composer = NAFPWaveformSliceComposer(
        input_key=input_key,
        output_key=output_key,
        shifted_key=shifted_key,
        duration=duration,
        offset_duration=0.2,
        sample_rate=sample_rate,
        training=True,
    )
    list_batch = composer(list_batch)
    list_batch = list(list_batch)

    for sample in list_batch:
        assert set(sample.keys()) == {
            "__key__",
            "waveform",
            "waveform_slice",
            "augmented_waveform_slice",
            "sample_rate",
        }

        assert sample["waveform_slice"].size() == (int(sample_rate * duration),)
        assert sample["augmented_waveform_slice"].size() == (int(sample_rate * duration),)

    composer = AudioFeatureExtractionComposer(
        aT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            f_min=30,
            f_max=8000,
            window_fn=torch.hann_window,
            n_mels=256,
        ),
        audio_key=output_key,
        feature_key=melspectrogram_key,
    )

    list_batch = composer(list_batch)

    composer = NAFPSpectrogramAugmentationComposer(
        input_key=melspectrogram_key,
        output_key=melspectrogram_key,
        training=True,
    )

    list_batch = composer(list_batch)

    for sample in list_batch:
        assert set(sample.keys()) == {
            "__key__",
            "waveform",
            "waveform_slice",
            "augmented_waveform_slice",
            "melspectrogram_slice",
            "sample_rate",
        }


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
