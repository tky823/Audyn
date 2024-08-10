"""This script is used to dump dummy MUSDB18 dataset."""

import os
from argparse import ArgumentParser, Namespace
from typing import Optional

import torch
import torchaudio
from torchaudio.io import StreamWriter
from tqdm import tqdm

from audyn.utils.data.musdb18 import (
    sources,
    test_track_names,
    train_track_names,
    validation_track_names,
)


def main() -> None:
    args = parse_args()

    root = args.root

    g = torch.Generator()
    g.manual_seed(0)

    num_channels = 2
    sample_rate = 4410
    duration = 1.5
    num_frames = int(duration * sample_rate)

    subset_name = "train"

    for track_name in tqdm(train_track_names):
        mp4_path = os.path.join(root, subset_name, f"{track_name}.stem.mp4")
        track_dir = os.path.join(root, subset_name, track_name)

        os.makedirs(track_dir, exist_ok=True)

        save_wav_files(track_dir, sample_rate, num_frames, num_channels=num_channels, generator=g)
        save_mp4_file(mp4_path, sample_rate, num_channels=num_channels)

    subset_name = "validation"

    for track_name in tqdm(validation_track_names):
        mp4_path = os.path.join(root, subset_name, f"{track_name}.stem.mp4")
        track_dir = os.path.join(root, subset_name, track_name)

        os.makedirs(track_dir, exist_ok=True)

        save_wav_files(track_dir, sample_rate, num_frames, num_channels=num_channels, generator=g)
        save_mp4_file(mp4_path, sample_rate, num_channels=num_channels)

    subset_name = "test"

    for track_name in tqdm(test_track_names):
        mp4_path = os.path.join(root, subset_name, f"{track_name}.stem.mp4")
        track_dir = os.path.join(root, subset_name, track_name)

        os.makedirs(track_dir, exist_ok=True)

        save_wav_files(track_dir, sample_rate, num_frames, num_channels=num_channels, generator=g)
        save_mp4_file(mp4_path, sample_rate, num_channels=num_channels)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Save dummy MUSDB18.")

    parser.add_argument(
        "--root",
        type=str,
        help="Path to save dummy MUSDB18 dataset.",
    )

    return parser.parse_args()


def save_wav_files(
    track_dir: str,
    sample_rate: int,
    num_frames: int,
    num_channels: int = 2,
    generator: Optional[torch.Generator] = None,
) -> None:
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(0)

    os.makedirs(track_dir, exist_ok=True)

    mixture = 0

    for source in sources:
        waveform = torch.randn((num_channels, num_frames), generator=generator)
        max_amplitude = torch.max(torch.abs(waveform))
        waveform = 0.1 * (waveform / max_amplitude)
        wav_path = os.path.join(track_dir, f"{source}.wav")
        torchaudio.save(wav_path, waveform, sample_rate)

        mixture = mixture + waveform

    wav_path = os.path.join(track_dir, "mixture.wav")
    torchaudio.save(wav_path, mixture, sample_rate)


def save_mp4_file(
    path: str,
    sample_rate: int,
    num_channels: int = 2,
) -> None:
    track_dir, _ = os.path.splitext(path)
    track_dir, _ = os.path.splitext(track_dir)
    writer = StreamWriter(path)

    for _ in ["mixture"] + sources:
        writer.add_audio_stream(
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

    with writer.open():
        for stream_idx, stream in enumerate(["mixture"] + sources):
            wav_path = os.path.join(track_dir, f"{stream}.wav")
            waveform, _ = torchaudio.load(wav_path, channels_first=False)
            writer.write_audio_chunk(stream_idx, waveform)


if __name__ == "__main__":
    main()
