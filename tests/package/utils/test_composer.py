from typing import Any, Dict

import pytest
import torch
import torchaudio.transforms as aT

from audyn.transforms.hubert import HuBERTMFCC
from audyn.utils.data.composer import AudioFeatureExtractionComposer, SequentialComposer


def test_hubert_composer(audioset_samples: Dict[str, Dict[str, Any]]) -> None:
    sample_rate = 16000
    list_batch = []

    for key, sample in audioset_samples.items():
        sample["__key__"] = key
        list_batch.append(sample)

    composer = AudioFeatureExtractionComposer(
        feature_extractor=HuBERTMFCC(sample_rate),
        audio_key="audio",
        feature_key="mfcc",
    )
    list_batch = composer(list_batch)

    for sample in list_batch:
        # HuBERTMFCC returns 39-dim feature per each frame.
        assert sample["mfcc"].size(0) == 39


def test_sequential_composer(audioset_samples: Dict[str, Dict[str, Any]]) -> None:
    sample_rate = 16000
    list_batch = []

    for key, sample in audioset_samples.items():
        sample["__key__"] = key
        list_batch.append(sample)

    hubert_mfcc_composer = AudioFeatureExtractionComposer(
        feature_extractor=HuBERTMFCC(sample_rate),
        audio_key="audio",
        feature_key="mfcc",
    )
    melspectrogram_composer = AudioFeatureExtractionComposer(
        feature_extractor=aT.MelSpectrogram(sample_rate=sample_rate),
        audio_key="audio",
        feature_key="melspec",
    )
    composer = SequentialComposer(hubert_mfcc_composer, melspectrogram_composer)

    list_batch = composer(list_batch)

    for sample in list_batch:
        # HuBERTMFCC returns 39-dim feature per each frame.
        assert sample["mfcc"].size(0) == 39


@pytest.fixture
def audioset_samples() -> Dict[str, Dict[str, Any]]:
    sample_rate = 44100

    g = torch.Generator()
    g.manual_seed(0)

    samples = {
        "example0": {
            "audio": torch.randn(3 * 44100, generator=g),
            "tags": ["/m/09x0r", "/m/05zppz"],
            "sample_rate": sample_rate,
        },
        "example1": {
            "audio": torch.randn(2 * 44100, generator=g),
            "tags": ["/m/02zsn"],
            "sample_rate": sample_rate,
        },
        "example2": {
            "audio": torch.randn(1 * 44100, generator=g),
            "tags": ["/m/05zppz", "/m/0ytgt", "/m/01h8n0", "/m/02qldy"],
            "sample_rate": sample_rate,
        },
        "example3": {
            "audio": torch.randn(1 * 44100, generator=g),
            "tags": ["/m/0261r1"],
            "sample_rate": sample_rate,
        },
        "example4": {
            "audio": torch.randn(2 * 44100, generator=g),
            "tags": ["/m/0261r1", "/m/09x0r", "/m/0brhx"],
            "sample_rate": sample_rate,
        },
        "example5": {
            "audio": torch.randn(1 * 44100, generator=g),
            "tags": ["/m/0261r1", "/m/07p6fty"],
            "sample_rate": sample_rate,
        },
        "example6": {
            "audio": torch.randn(3 * 44100, generator=g),
            "tags": ["/m/09x0r", "/m/07q4ntr"],
            "sample_rate": sample_rate,
        },
        "example7": {
            "audio": torch.randn(1 * 44100, generator=g),
            "tags": ["/m/09x0r", "/m/07rwj3x", "/m/07sr1lc"],
            "sample_rate": sample_rate,
        },
        "example8": {
            "audio": torch.randn(3 * 44100, generator=g),
            "tags": ["/m/04gy_2"],
            "sample_rate": sample_rate,
        },
        "example9": {
            "audio": torch.randn(1 * 44100, generator=g),
            "tags": ["/t/dd00135", "/m/03qc9zr", "/m/02rtxlg", "/m/01j3sz"],
            "sample_rate": sample_rate,
        },
        "example10": {
            "audio": torch.randn(2 * 44100, generator=g),
            "tags": ["/m/0261r1", "/m/05zppz", "/t/dd00001"],
            "sample_rate": sample_rate,
        },
    }

    return samples
