from typing import Any, Dict

import pytest
import torch
import torchaudio.transforms as aT

from audyn.transforms.clap import (
    LAIONAudioEncoder2023MelSpectrogram,
    LAIONAudioEncoder2023MelSpectrogramFusion,
)
from audyn.transforms.hubert import HuBERTMFCC
from audyn.transforms.slicer import WaveformSlicer
from audyn.utils.data.audioset import num_tags as num_audioset_tags
from audyn.utils.data.audioset import tags as audioset_tags
from audyn.utils.data.clap.composer import LAIONAudioEncoder2023Composer
from audyn.utils.data.composer import (
    AudioFeatureExtractionComposer,
    LabelsToMultihot,
    LabelToOnehot,
    SequentialComposer,
    SynchronousWaveformSlicer,
)
from audyn.utils.data.gtzan import num_tags as num_gtzan_tags
from audyn.utils.data.gtzan import tags as gtzan_tags
from audyn.utils.data.hifigan.composer import HiFiGANComposer


def test_synchronous_waveform_slicer(audioset_samples: Dict[str, Dict[str, Any]]) -> None:
    sample_rate = 16000
    list_batch = []

    for key, sample in audioset_samples.items():
        sample["__key__"] = key
        list_batch.append(sample)

    composer = SynchronousWaveformSlicer(
        input_keys=["audio", "audio"],
        output_keys=["audio_slice1", "audio_slice2"],
        length=int(1.5 * sample_rate),
        training=True,
    )
    list_batch = composer(list_batch)

    for sample in list_batch:
        assert torch.equal(sample["audio_slice1"], sample["audio_slice2"])


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


def test_label_to_onehot(gtzan_samples: Dict[str, Dict[str, Any]]) -> None:
    sample_rate = 16000
    list_batch = []

    for key, sample in gtzan_samples.items():
        sample["__key__"] = key
        list_batch.append(sample)

    melspectrogram_composer = AudioFeatureExtractionComposer(
        feature_extractor=aT.MelSpectrogram(sample_rate=sample_rate),
        audio_key="audio",
        feature_key="melspec",
    )
    labels_to_onehot = LabelToOnehot(
        label_key="tag",
        feature_key="onehot",
        labels=gtzan_tags,
    )
    composer = SequentialComposer(
        melspectrogram_composer,
        labels_to_onehot,
    )

    list_batch = composer(list_batch)

    for sample in list_batch:
        assert sample["onehot"].size() == (num_gtzan_tags,)


def test_labels_to_multihot(audioset_samples: Dict[str, Dict[str, Any]]) -> None:
    sample_rate = 44100
    tags = [tag["tag"] for tag in audioset_tags]
    list_batch = []

    for key, sample in audioset_samples.items():
        sample["__key__"] = key
        list_batch.append(sample)

    melspectrogram_composer = AudioFeatureExtractionComposer(
        feature_extractor=aT.MelSpectrogram(sample_rate=sample_rate),
        audio_key="audio",
        feature_key="melspec",
    )
    labels_to_multihot = LabelsToMultihot(
        label_key="tags",
        feature_key="multihot",
        labels=tags,
    )
    composer = SequentialComposer(
        melspectrogram_composer,
        labels_to_multihot,
    )

    list_batch = composer(list_batch)

    for sample in list_batch:
        assert sample["multihot"].size() == (num_audioset_tags,)


def test_hifigan_composer(audioset_samples: Dict[str, Dict[str, Any]]) -> None:
    audio_key = "audio"
    sample_rate = 16000
    n_mels = 80
    length = 8192
    list_batch = []

    melspectrogram_transform = aT.MelSpectrogram(sample_rate, n_mels=n_mels)
    waveform_slicer = WaveformSlicer(length=length)

    for key, sample in audioset_samples.items():
        sample["__key__"] = key
        list_batch.append(sample)

    composer = HiFiGANComposer(
        melspectrogram_transform=melspectrogram_transform,
        slicer=waveform_slicer,
        waveform_key=audio_key,
    )
    list_batch = composer(list_batch)

    for sample in list_batch:
        assert {
            audio_key,
            "sample_rate",
            "melspectrogram",
            "waveform_slice",
            "melspectrogram_slice",
        } == set(sample.keys())
        assert sample["waveform_slice"].size(-1) == length


def test_laion_clap_composer(audioset_samples: Dict[str, Dict[str, Any]]) -> None:
    audio_key = "audio"
    sample_rate = 48000
    n_mels = 64
    chunk_size = 301
    num_chunks = 3
    list_batch = []

    melspectrogram_transform = LAIONAudioEncoder2023MelSpectrogram(sample_rate, n_mels=n_mels)
    fusion_transform = LAIONAudioEncoder2023MelSpectrogramFusion(
        chunk_size=chunk_size, num_chunks=num_chunks
    )

    for key, sample in audioset_samples.items():
        sample["__key__"] = key
        list_batch.append(sample)

    composer = LAIONAudioEncoder2023Composer(
        melspectrogram_transform=melspectrogram_transform,
        fusion_transform=fusion_transform,
        waveform_key=audio_key,
    )
    list_batch = composer(list_batch)

    for sample in list_batch:
        assert {
            audio_key,
            "sample_rate",
            "melspectrogram",
            "fused_melspectrogram",
        } == set(sample.keys())

        assert sample["fused_melspectrogram"].size() == (num_chunks + 1, n_mels, chunk_size)


@pytest.fixture
def audioset_samples() -> Dict[str, Dict[str, Any]]:
    sample_rate = 16000

    g = torch.Generator()
    g.manual_seed(0)

    def _create_dummy_audio(duration: float) -> torch.Tensor:
        return torch.randn(int(duration * sample_rate), generator=g)

    samples = {
        "example0": {
            "audio": _create_dummy_audio(3),
            "tags": ["/m/09x0r", "/m/05zppz"],
            "sample_rate": sample_rate,
        },
        "example1": {
            "audio": _create_dummy_audio(2),
            "tags": ["/m/02zsn"],
            "sample_rate": sample_rate,
        },
        "example2": {
            "audio": _create_dummy_audio(1),
            "tags": ["/m/05zppz", "/m/0ytgt", "/m/01h8n0", "/m/02qldy"],
            "sample_rate": sample_rate,
        },
        "example3": {
            "audio": _create_dummy_audio(1),
            "tags": ["/m/0261r1"],
            "sample_rate": sample_rate,
        },
        "example4": {
            "audio": _create_dummy_audio(1),
            "tags": ["/m/0261r1", "/m/09x0r", "/m/0brhx"],
            "sample_rate": sample_rate,
        },
        "example5": {
            "audio": _create_dummy_audio(1),
            "tags": ["/m/0261r1", "/m/07p6fty"],
            "sample_rate": sample_rate,
        },
        "example6": {
            "audio": _create_dummy_audio(3),
            "tags": ["/m/09x0r", "/m/07q4ntr"],
            "sample_rate": sample_rate,
        },
        "example7": {
            "audio": _create_dummy_audio(1),
            "tags": ["/m/09x0r", "/m/07rwj3x", "/m/07sr1lc"],
            "sample_rate": sample_rate,
        },
        "example8": {
            "audio": _create_dummy_audio(3),
            "tags": ["/m/04gy_2"],
            "sample_rate": sample_rate,
        },
        "example9": {
            "audio": _create_dummy_audio(1),
            "tags": ["/t/dd00135", "/m/03qc9zr", "/m/02rtxlg", "/m/01j3sz"],
            "sample_rate": sample_rate,
        },
        "example10": {
            "audio": _create_dummy_audio(2),
            "tags": ["/m/0261r1", "/m/05zppz", "/t/dd00001"],
            "sample_rate": sample_rate,
        },
    }

    return samples


@pytest.fixture
def gtzan_samples() -> Dict[str, Dict[str, Any]]:
    duration = 30
    sample_rate = 16000

    g = torch.Generator()
    g.manual_seed(0)

    def _create_dummy_audio() -> torch.Tensor:
        return torch.randn(int(duration * sample_rate), generator=g)

    samples = {
        "example0": {
            "audio": _create_dummy_audio(),
            "tag": "blues",
            "sample_rate": sample_rate,
        },
        "example1": {
            "audio": _create_dummy_audio(),
            "tag": "classical",
            "sample_rate": sample_rate,
        },
        "example2": {
            "audio": _create_dummy_audio(),
            "tag": "country",
            "sample_rate": sample_rate,
        },
        "example3": {
            "audio": _create_dummy_audio(),
            "tag": "disco",
            "sample_rate": sample_rate,
        },
        "example4": {
            "audio": _create_dummy_audio(),
            "tag": "hiphop",
            "sample_rate": sample_rate,
        },
        "example5": {
            "audio": _create_dummy_audio(),
            "tag": "jazz",
            "sample_rate": sample_rate,
        },
        "example6": {
            "audio": _create_dummy_audio(),
            "tag": "reggae",
            "sample_rate": sample_rate,
        },
        "example7": {
            "audio": _create_dummy_audio(),
            "tag": "rock",
            "sample_rate": sample_rate,
        },
        "example8": {
            "audio": _create_dummy_audio(),
            "tag": "metal",
            "sample_rate": sample_rate,
        },
        "example9": {
            "audio": _create_dummy_audio(),
            "tag": "pop",
            "sample_rate": sample_rate,
        },
        "example10": {
            "audio": _create_dummy_audio(),
            "tag": "blues",
            "sample_rate": sample_rate,
        },
    }

    return samples
