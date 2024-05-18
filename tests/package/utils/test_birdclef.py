import os
import tempfile
from typing import Any, Dict

import pytest
import torch
import torchaudio
import webdataset as wds
from torch.utils.data import DataLoader

from audyn.utils.data import WebDatasetWrapper
from audyn.utils.data.birdclef.birdclef2024.composer import BirdCLEF2024PrimaryLabelComposer
from audyn.utils.data.collator import Collator


@pytest.mark.parametrize("composer_pattern", [1, 2])
def test_birdclef2024_primary_label_composer(
    birdclef2024_samples: Dict[str, Dict[str, Any]], composer_pattern: str
) -> None:
    torch.manual_seed(0)

    max_shard_count = 4
    audio_key, sample_rate_key, filename_key = "audio", "sample_rate", "filename"
    label_name_key, label_index_key = "primary_label", "label_index"

    batch_size = 3

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        audio_dir = os.path.join(temp_dir, "audio")
        list_dir = os.path.join(temp_dir, "list")
        feature_dir = os.path.join(temp_dir, "feature")

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(list_dir, exist_ok=True)
        os.makedirs(feature_dir, exist_ok=True)

        list_path = os.path.join(list_dir, "train.txt")
        tar_path = os.path.join(feature_dir, "%d.tar")

        try:
            # save dummy audio to check usage of .ogg file.
            dummy_sample_rate = 16000
            waveform = torch.randn((2, 10 * dummy_sample_rate))
            amplitude = torch.abs(waveform)
            waveform = waveform / torch.max(amplitude)
            waveform = 0.9 * waveform
            path = os.path.join(audio_dir, "audio.ogg")
            torchaudio.save(path, waveform, dummy_sample_rate)
            is_ogg_supported = True
        except RuntimeError:
            is_ogg_supported = False

        if is_ogg_supported:
            with wds.ShardWriter(tar_path, maxcount=max_shard_count) as sink, open(
                list_path, mode="w"
            ) as f_list:
                for filename in sorted(birdclef2024_samples.keys()):
                    sample = birdclef2024_samples[filename]
                    sample_rate = sample["sample_rate"]
                    primary_label = sample["primary_label"]
                    waveform = torch.randn((2, 10 * sample_rate))
                    amplitude = torch.abs(waveform)
                    waveform = waveform / torch.max(amplitude)
                    waveform = 0.9 * waveform
                    path = os.path.join(audio_dir, f"{filename}.ogg")
                    torchaudio.save(path, waveform, sample_rate)

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
                len(os.listdir(feature_dir))
                == (len(birdclef2024_samples) - 1) // max_shard_count + 1
            )

            composer = BirdCLEF2024PrimaryLabelComposer(
                label_name_key=label_name_key,
                label_index_key=label_index_key,
            )

            if composer_pattern == 1:
                # pattern 1: set composer to dataset
                collator = Collator()
                dataset = WebDatasetWrapper.instantiate_dataset(
                    list_path,
                    feature_dir,
                    composer=composer,
                )
            elif composer_pattern == 2:
                # pattern 2: set composer to collator
                collator = Collator(composer=composer)
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
                    "__key__",
                    "__url__",
                    "audio",
                    "sample_rate",
                    "primary_label",
                    "filename",
                    "label_index",
                }


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
