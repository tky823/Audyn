import glob
import os
import tempfile

import pytest
import torch
import torchaudio
from dummy import allclose
from torch.utils.data import DataLoader

from audyn.utils.data.musdb18 import (
    sources,
    test_track_names,
    train_track_names,
    validation_track_names,
)
from audyn.utils.data.musdb18.dataset import (
    MUSDB18,
    RandomStemsMUSDB18Dataset,
    StemsMUSDB18Dataset,
)


def test_musdb18() -> None:
    num_frames = 48000

    with tempfile.TemporaryDirectory() as temp_dir:
        _save_dummy_musdb18(root=temp_dir, num_frames=num_frames)

        dataset = MUSDB18(temp_dir, subset="train", ext="wav")

        for track in dataset:
            track.frame_offset = num_frames // 4
            track.num_frames = num_frames // 2

            drums, _ = track.drums
            bass, _ = track.bass
            other, _ = track.other
            vocals, _ = track.vocals
            accompaniment, _ = track.accompaniment
            mixture, _ = track.mixture
            stems, _ = track.stems

            assert track.name in train_track_names
            allclose(drums + bass + other + vocals, mixture, atol=1e-4)
            allclose(accompaniment + vocals, mixture, atol=1e-4)
            allclose(stems[0], mixture, atol=1e-4)
            allclose(stems[1], drums)
            allclose(stems[2], bass)
            allclose(stems[3], other)
            allclose(stems[4], vocals)

            break


@pytest.mark.parametrize("replacement", [True, False])
def test_musdb18_dataset(replacement: bool) -> None:
    batch_size = 3
    num_workers = 2
    num_frames = 48000
    duration = 1.0

    with tempfile.TemporaryDirectory() as temp_dir:
        feature_dir = os.path.join(temp_dir, "features")
        list_dir = os.path.join(temp_dir, "list")
        train_feature_dir = os.path.join(feature_dir, "train")
        train_list_path = os.path.join(list_dir, "train.txt")

        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(list_dir, exist_ok=True)

        _save_dummy_musdb18(root=feature_dir, num_frames=num_frames)

        with open(train_list_path, mode="w") as f:
            for track_name in sorted(glob.glob(os.path.join(train_feature_dir, "*"))):
                f.write(track_name + "\n")

        if not replacement:
            # StemsMUSDB18Dataset
            dataset = StemsMUSDB18Dataset(train_list_path, train_feature_dir, duration=duration)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

            filenames = []

            for feature in dataloader:
                assert set(feature.keys()) == {
                    "drums",
                    "bass",
                    "other",
                    "vocals",
                    "sample_rate",
                    "filename",
                }

                filenames_per_batch = set()

                for filename in feature["filename"]:
                    # If ``replacement=True``, filename may be included in filenames_per_batch
                    # with tiny probability.
                    assert filename not in filenames_per_batch

                    filenames_per_batch.add(filename)
                    filenames.append(filename)

            assert len(set(filenames)) == len(dataset.filenames)

        # RandomStemsMUSDB18Dataset
        dataset = RandomStemsMUSDB18Dataset(
            train_list_path, train_feature_dir, duration=duration, replacement=replacement
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        filenames = []

        for feature in dataloader:
            assert set(feature.keys()) == {
                "drums",
                "bass",
                "other",
                "vocals",
                "sample_rate",
                "filename",
            }

            filenames_per_batch = set()

            for filename in feature["filename"]:
                # If ``replacement=True``, filename may be included in filenames_per_batch
                # with tiny probability.
                assert filename not in filenames_per_batch

                filenames_per_batch.add(filename)
                filenames.append(filename)

        if not replacement:
            assert len(set(filenames)) == len(dataset.filenames)


def _save_dummy_musdb18(root: str, num_frames: int) -> None:
    g = torch.Generator()
    g.manual_seed(0)

    num_channels = 2
    sample_rate = 24000

    subset_name = "train"

    for track_name in train_track_names:
        track_dir = os.path.join(root, subset_name, track_name)

        os.makedirs(track_dir, exist_ok=True)

        mixture = 0

        for source in sources:
            path = os.path.join(track_dir, f"{source}.wav")
            waveform = torch.randn((num_channels, num_frames), generator=g)
            max_amplitude = torch.max(torch.abs(waveform))
            waveform = 0.1 * (waveform / max_amplitude)
            mixture = mixture + waveform

            torchaudio.save(path, waveform, sample_rate)

        path = os.path.join(track_dir, "mixture.wav")
        torchaudio.save(path, mixture, sample_rate)

    for track_name in validation_track_names:
        track_dir = os.path.join(root, subset_name, track_name)

        os.makedirs(track_dir, exist_ok=True)

        mixture = 0

        for source in sources:
            path = os.path.join(track_dir, f"{source}.wav")
            waveform = torch.randn((num_channels, num_frames), generator=g)
            max_amplitude = torch.max(torch.abs(waveform))
            waveform = 0.1 * (waveform / max_amplitude)
            mixture = mixture + waveform

            torchaudio.save(path, waveform, sample_rate)

        path = os.path.join(track_dir, "mixture.wav")
        torchaudio.save(path, mixture, sample_rate)

    subset_name = "test"

    for track_name in test_track_names:
        track_dir = os.path.join(root, subset_name, track_name)

        os.makedirs(track_dir, exist_ok=True)

        mixture = 0

        for source in sources:
            path = os.path.join(track_dir, f"{source}.wav")
            waveform = torch.randn((num_channels, num_frames), generator=g)
            max_amplitude = torch.max(torch.abs(waveform))
            waveform = 0.1 * (waveform / max_amplitude)
            mixture = mixture + waveform

            torchaudio.save(path, waveform, sample_rate)

        path = os.path.join(track_dir, "mixture.wav")
        torchaudio.save(path, mixture, sample_rate)
