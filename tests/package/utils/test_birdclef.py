import os
import sys
import tempfile
from typing import Any, Dict

import pytest
import torch
import torchaudio
import webdataset as wds
from torch.utils.data import DataLoader

from audyn.transforms.birdclef import BirdCLEF2024BaselineMelSpectrogram
from audyn.utils.data import WebDatasetWrapper
from audyn.utils.data.birdclef.birdclef2024 import decode_csv_line
from audyn.utils.data.birdclef.birdclef2024 import (
    num_primary_labels as num_birdclef2024_primary_labels,
)
from audyn.utils.data.birdclef.birdclef2024.collator import BirdCLEF2024BaselineCollator
from audyn.utils.data.birdclef.birdclef2024.composer import (
    BirdCLEF2024PrimaryLabelComposer,
)
from audyn.utils.github import download_file_from_github_release

IS_WINDOWS = sys.platform == "win32"


def test_decode_csv_line() -> None:
    line = (
        "primary",
        "[]",
        "['call']",
        "1.1",
        "2.2",
        "scientific name",
        "common name",
        "author",
        "license",
        "1.5",
        "https://example.com/001",
        "primary/example.ogg",
    )
    data = decode_csv_line(line)

    assert set(data.keys()) == {
        "filename",
        "primary_label",
        "secondary_label",
        "type",
        "latitude",
        "longitude",
        "scientific_name",
        "common_name",
        "rating",
        "path",
    }


@pytest.mark.parametrize("composer_pattern", [1, 2])
def test_birdclef2024_primary_label_composer(
    birdclef2024_samples: Dict[str, Dict[str, Any]], composer_pattern: str
) -> None:
    torch.manual_seed(0)

    try:
        from torchvision.transforms.v2 import MixUp  # noqa: F401
    except ImportError:
        import torchvision

        pytest.skip(f"MixUp is not supported by torchvision={torchvision.__version__}.")

    max_shard_count = 4
    audio_key, sample_rate_key, filename_key = "audio", "sample_rate", "filename"
    waveform_key, melspectrogram_key = "waveform", "melspectrogram"
    label_name_key, label_index_key = "primary_label", "label_index"
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.1.dev4/piano-48k.ogg"

    melspectrogram_transform = BirdCLEF2024BaselineMelSpectrogram()

    batch_size = 3

    if IS_WINDOWS:
        pytest.skip(".ogg file is not supported by Windows.")

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        # save dummy audio to check usage of .ogg file.
        audio_dir = os.path.join(temp_dir, "audio")
        path = os.path.join(audio_dir, "audio.ogg")
        download_file_from_github_release(url, path)

        try:
            waveform, sample_rate = torchaudio.load(path)
            is_ogg_supported = True
        except RuntimeError:
            is_ogg_supported = False

    if not is_ogg_supported:
        pytest.skip(".ogg file is not supported by environment.")

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        audio_dir = os.path.join(temp_dir, "audio")
        list_dir = os.path.join(temp_dir, "list")
        feature_dir = os.path.join(temp_dir, "feature")

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(list_dir, exist_ok=True)
        os.makedirs(feature_dir, exist_ok=True)

        list_path = os.path.join(list_dir, "train.txt")
        tar_path = os.path.join(feature_dir, "%d.tar")

        audio_dir = os.path.join(temp_dir, "audio")
        path = os.path.join(audio_dir, "audio.ogg")
        download_file_from_github_release(url, path)

        with (
            wds.ShardWriter(tar_path, maxcount=max_shard_count) as sink,
            open(list_path, mode="w") as f_list,
        ):
            for filename in sorted(birdclef2024_samples.keys()):
                sample = birdclef2024_samples[filename]
                sample_rate = sample["sample_rate"]
                primary_label = sample["primary_label"]

                with open(path, mode="rb") as f_audio:
                    audio = f_audio.read()

                feature = {
                    "__key__": filename,
                    f"{audio_key}.ogg": audio,
                    f"{label_name_key}.txt": primary_label,
                    f"{filename_key}.txt": filename,
                    f"{sample_rate_key}.pth": torch.tensor(sample_rate, dtype=torch.long),
                }

                sink.write(feature)
                f_list.write(filename + "\n")

        assert (
            len(os.listdir(feature_dir)) == (len(birdclef2024_samples) - 1) // max_shard_count + 1
        )

        composer = BirdCLEF2024PrimaryLabelComposer(
            melspectrogram_transform,
            audio_key=audio_key,
            sample_rate_key=sample_rate_key,
            label_name_key=label_name_key,
            filename_key=filename_key,
            waveform_key=waveform_key,
            melspectrogram_key=melspectrogram_key,
            label_index_key=label_index_key,
        )

        if composer_pattern == 1:
            # pattern 1: set composer to dataset
            collator = BirdCLEF2024BaselineCollator(
                melspectrogram_key=melspectrogram_key,
                label_index_key=label_index_key,
            )
            dataset = WebDatasetWrapper.instantiate_dataset(
                list_path,
                feature_dir,
                composer=composer,
            )
        elif composer_pattern == 2:
            # pattern 2: set composer to collator
            collator = BirdCLEF2024BaselineCollator(
                composer=composer,
                melspectrogram_key=melspectrogram_key,
                label_index_key=label_index_key,
            )
            dataset = WebDatasetWrapper.instantiate_dataset(
                list_path,
                feature_dir,
            )
        else:
            raise ValueError(f"Invalid composer_pattern {composer_pattern} is given.")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
        )

        for batch in dataloader:
            assert set(batch.keys()) == {
                waveform_key,
                melspectrogram_key,
                label_index_key,
                filename_key,
            }

            assert batch[label_index_key].size(-1) == num_birdclef2024_primary_labels


@pytest.fixture
def birdclef2024_samples() -> Dict[str, Dict[str, Any]]:
    sample_rate = 32000

    samples = {
        "example0": {
            "primary_label": "asbfly",
            "sample_rate": sample_rate,
        },
        "example1": {
            "primary_label": "ashdro1",
            "sample_rate": sample_rate,
        },
        "example2": {
            "primary_label": "ashpri1",
            "sample_rate": sample_rate,
        },
        "example3": {
            "primary_label": "ashpri1",
            "sample_rate": sample_rate,
        },
        "example4": {
            "primary_label": "ashdro1",
            "sample_rate": sample_rate,
        },
        "example5": {
            "primary_label": "asbfly",
            "sample_rate": sample_rate,
        },
        "example6": {
            "primary_label": "asbfly",
            "sample_rate": sample_rate,
        },
        "example7": {
            "primary_label": "ashpri1",
            "sample_rate": sample_rate,
        },
        "example8": {
            "primary_label": "ashdro1",
            "sample_rate": sample_rate,
        },
        "example9": {
            "primary_label": "ashpri1",
            "sample_rate": sample_rate,
        },
    }

    return samples
