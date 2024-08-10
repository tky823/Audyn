"""This script is used to dump dummy MUSDB18 dataset."""

import os
from argparse import ArgumentParser, Namespace
from typing import Optional

import torch
from torchaudio.io import StreamReader, StreamWriter
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
    sample_rate = 44100
    duration = 1.0
    num_frames = int(duration * sample_rate)

    subset_name = "train"

    for track_name in tqdm(train_track_names):
        mp4_path = os.path.join(root, subset_name, f"{track_name}.stem.mp4")
        track_dir = os.path.join(root, subset_name, track_name)

        os.makedirs(track_dir, exist_ok=True)

        save_mp4_file(mp4_path, sample_rate, num_frames, num_channels=num_channels, generator=g)
        save_wav_files(track_dir)

    subset_name = "validation"

    for track_name in tqdm(validation_track_names):
        mp4_path = os.path.join(root, subset_name, f"{track_name}.stem.mp4")
        track_dir = os.path.join(root, subset_name, track_name)

        os.makedirs(track_dir, exist_ok=True)

        save_mp4_file(mp4_path, sample_rate, num_frames, num_channels=num_channels, generator=g)
        save_wav_files(track_dir)

    subset_name = "test"

    for track_name in tqdm(test_track_names):
        mp4_path = os.path.join(root, subset_name, f"{track_name}.stem.mp4")
        track_dir = os.path.join(root, subset_name, track_name)

        os.makedirs(track_dir, exist_ok=True)

        save_mp4_file(mp4_path, sample_rate, num_frames, num_channels=num_channels, generator=g)
        save_wav_files(track_dir)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Save dummy MUSDB18.")

    parser.add_argument(
        "--root",
        type=str,
        help="Path to save dummy MUSDB18 dataset.",
    )

    return parser.parse_args()


def save_wav_files(track_dir: str) -> None:
    mp4_path = f"{track_dir}.stem.mp4"

    for stream_idx, source in enumerate(["mixture"] + sources):
        wav_path = os.path.join(track_dir, f"{source}.wav")
        reader = StreamReader(mp4_path)
        writer = StreamWriter(wav_path)

        stream_info = reader.get_src_stream_info(stream_idx)
        sample_rate = stream_info.sample_rate
        num_channels = stream_info.num_channels

        assert sample_rate == 44100

        sample_rate = int(sample_rate)

        reader.add_audio_stream(
            frames_per_chunk=sample_rate,
            stream_index=stream_idx,
        )
        writer.add_audio_stream(
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        with writer.open():
            for (chunk,) in reader.stream():
                writer.write_audio_chunk(0, chunk)


def save_mp4_file(
    path: str,
    sample_rate: int,
    num_frames: int,
    num_channels: int = 2,
    generator: Optional[torch.Generator] = None,
) -> None:
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(0)

    writer = StreamWriter(path)

    mixture = 0
    waveforms = {}

    for source in sources:
        waveform = torch.randn((num_frames, num_channels), generator=generator)
        max_amplitude = torch.max(torch.abs(waveform))
        waveform = 0.1 * (waveform / max_amplitude)
        waveforms[source] = waveform
        mixture = mixture + waveform

    waveforms["mixture"] = mixture

    for _ in ["mixture"] + sources:
        writer.add_audio_stream(
            sample_rate=sample_rate,
            num_channels=num_channels,
            encoder="aac",
            encoder_sample_rate=sample_rate,
            encoder_num_channels=num_channels,
        )

    with writer.open():
        for stream_idx, stream in enumerate(["mixture"] + sources):
            waveform = waveforms[stream]
            writer.write_audio_chunk(stream_idx, waveform)


if __name__ == "__main__":
    main()
