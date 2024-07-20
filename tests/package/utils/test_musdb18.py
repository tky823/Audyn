import os
import tempfile

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
from audyn.utils.data.musdb18.dataset import MUSDB18, RandomStemsMUSDB18Dataset


def test_musdb18() -> None:
    num_frames = 48000

    with tempfile.TemporaryDirectory() as temp_dir:
        _save_dummy_musdb18(root=temp_dir, num_frames=num_frames)
        dataset = MUSDB18(temp_dir, subset="train")

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


def test_musdb18_dataset() -> None:
    batch_size = 4
    num_workers = 2
    num_frames = 48000

    with tempfile.TemporaryDirectory() as temp_dir:
        _save_dummy_musdb18(root=temp_dir, num_frames=num_frames)

        dataset = RandomStemsMUSDB18Dataset(temp_dir, subset="train", duration=1.0)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        for feature in dataloader:
            assert set(feature.keys()) == {
                "drums",
                "bass",
                "other",
                "vocals",
                "sample_rate",
                "filename",
            }


def _save_dummy_musdb18(root: str, num_frames: int) -> None:
    g = torch.Generator()
    g.manual_seed(0)

    num_channels = 2
    sample_rate = 24000

    subset = "train"

    for track_name in train_track_names:
        track_dir = os.path.join(root, subset, track_name)

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
        track_dir = os.path.join(root, subset, track_name)

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

    subset = "test"

    for track_name in test_track_names:
        track_dir = os.path.join(root, subset, track_name)

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
