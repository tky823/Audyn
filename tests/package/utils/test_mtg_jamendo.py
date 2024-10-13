import os
import tempfile
from typing import Any, Dict, List

import pytest
import torch
import torchaudio
import webdataset as wds

from audyn.utils.data import WebDatasetWrapper, WebLoaderWrapper
from audyn.utils.data.collator import Collator
from audyn.utils.data.mtg_jamendo import (
    MTGJamendoEvaluationWaveformSliceComposer,
    download_all_metadata,
    download_genre_metadata,
    download_instrument_metadata,
    download_moodtheme_metadata,
    download_top50_metadata,
    genre_tags,
    instrument_tags,
    moodtheme_tags,
    num_genre_tags,
    num_instrument_tags,
    num_moodtheme_tags,
    num_top50_tags,
    top50_tags,
)


def test_mtg_jamendo() -> None:
    top50_metadata = download_top50_metadata()
    genre_metadata = download_genre_metadata()
    instrument_metadata = download_instrument_metadata()
    moodtheme_metadata = download_moodtheme_metadata()

    assert len(top50_tags) == num_top50_tags
    assert len(genre_tags) == num_genre_tags
    assert len(instrument_tags) == num_instrument_tags
    assert len(moodtheme_tags) == num_moodtheme_tags

    sample = top50_metadata[15000]

    assert sample["track"] == "track_1031094"
    assert sample["artist"] == "artist_433491"
    assert sample["album"] == "album_121295"
    assert sample["path"] == "94/1031094.mp3"
    assert sample["duration"] == 198.6
    assert sample["tags"] == [
        "genre---electronic",
        "genre---experimental",
        "instrument---piano",
        "instrument---synthesizer",
    ]

    sample = genre_metadata[0]

    assert sample["track"] == "track_0000241"
    assert sample["artist"] == "artist_000005"
    assert sample["album"] == "album_000033"
    assert sample["path"] == "41/241.mp3"
    assert sample["duration"] == 340.1
    assert sample["tags"] == ["genre---rock"]

    sample = instrument_metadata[0]

    assert sample["track"] == "track_0000382"
    assert sample["artist"] == "artist_000020"
    assert sample["album"] == "album_000046"
    assert sample["path"] == "82/382.mp3"
    assert sample["duration"] == 211.1
    assert sample["tags"] == ["instrument---voice"]

    sample = moodtheme_metadata[0]

    assert sample["track"] == "track_0000948"
    assert sample["artist"] == "artist_000087"
    assert sample["album"] == "album_000149"
    assert sample["path"] == "48/948.mp3"
    assert sample["duration"] == 212.7
    assert sample["tags"] == ["mood/theme---background"]


def test_mtg_jamendo_composer(mtg_jamendo_samples: List[Dict[str, Any]]) -> None:
    sample_rate = 22050
    seed = 0

    torch.manual_seed(seed)

    max_shard_count = 4
    audio_key, waveform_key = "audio", "waveform"
    batch_size = 5

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_dir = os.path.join(temp_dir, "audio")
        list_dir = os.path.join(temp_dir, "list")
        feature_dir = os.path.join(temp_dir, "feature")

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(list_dir, exist_ok=True)
        os.makedirs(feature_dir, exist_ok=True)

        list_path = os.path.join(list_dir, "train.txt")
        tar_path = os.path.join(feature_dir, "%d.tar")

        with (
            wds.ShardWriter(tar_path, maxcount=max_shard_count) as sink,
            open(list_path, mode="w") as f_list,
        ):
            for sample in mtg_jamendo_samples:
                album = sample["album"]
                artist = sample["artist"]
                track = sample["track"]
                duration = sample["duration"]
                path = sample["path"]
                tags = sample["tags"]

                filename = path.replace(".mp3", "")
                waveform = torch.randn((2, int(duration * sample_rate)))
                amplitude = torch.abs(waveform)
                waveform = waveform / torch.max(amplitude)
                waveform = 0.9 * waveform
                path = os.path.join(audio_dir, f"{track}.wav")
                torchaudio.save(path, waveform, sample_rate)

                with open(path, mode="rb") as f_audio:
                    audio = f_audio.read()

                feature = {
                    "__key__": filename,
                    "track.txt": track,
                    "album.txt": album,
                    "artist.txt": artist,
                    "duration.pth": torch.tensor(duration, dtype=torch.float),
                    "audio.wav": audio,
                    "sample_rate.pth": torch.tensor(sample_rate, dtype=torch.long),
                    "tags.json": tags,
                    "filename.txt": filename,
                }

                sink.write(feature)
                f_list.write(filename + "\n")

        assert (
            len(os.listdir(feature_dir)) == (len(mtg_jamendo_samples) - 1) // max_shard_count + 1
        )

        composer = MTGJamendoEvaluationWaveformSliceComposer(
            audio_key, waveform_key, duration=10, sample_rate=22050, num_slices=8
        )
        collator = Collator(composer=composer)
        dataset = WebDatasetWrapper.instantiate_dataset(list_path, feature_dir)
        dataloader = WebLoaderWrapper.instantiate_dataloader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
        )

        samples_per_epoch = 0

        for sample in dataloader:
            samples_per_epoch += len(sample["filename"])

        assert samples_per_epoch == len(mtg_jamendo_samples)


@pytest.fixture
def mtg_jamendo_samples() -> List[Dict[str, Any]]:
    samples = download_all_metadata()
    samples = samples[::5000]

    return samples
