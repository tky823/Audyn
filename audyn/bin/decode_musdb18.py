import glob
import os
import shutil
import tempfile
from typing import Optional

from omegaconf import DictConfig
from torchaudio.io import StreamReader, StreamWriter

from ..utils.data.musdb18 import (
    sources,
    test_track_names,
    train_track_names,
    validation_track_names,
)
from ..utils.hydra import main as audyn_main

try:
    from tqdm import tqdm

    IS_TQDM_AVAILABLE = True
except ImportError:
    IS_TQDM_AVAILABLE = False


@audyn_main(config_name="decode-musdb18")
def main(config: DictConfig) -> None:
    r"""Decode .stem.mp4 file(s) into .wav files for MUSDB18 dataset.

    .. code-block:: shell

        track_name="ANiMAL - Rockshow"

        # "ANiMAL - Rockshow" is treated as validation split, but included in train/ folder.
        subset="validation"

        # case 1: decode single .stem.mp4 file
        # Each source name (drums, bass, etc.) is assigned to SOURCE in ``wav_path``.
        musdb18_mp4_path="${musdb18_mp4_root}/train/${track_name}.stem.mp4"
        musdb18_wav_path="${musdb18_wav_root}/train/${track_name}/SOURCE.wav"

        audyn-decode-musdb18 \
        mp4_path="${musdb18_mp4_path}"
        wav_path="${musdb18_wav_path}"

        # case 2: decode multiple .stem.mp4 files under specified directory
        musdb18_mp4_root="./MUSDB18"
        musdb18_wav_root="./MUSDB18-wav"

        audyn-decode-musdb18 \
        mp4_root="${musdb18_mp4_root}/train" \
        wav_root="${musdb18_wav_root}/train"

        # case 3: decode multiple .stem.mp4 files for specified subset (train, validation, or test)
        audyn-decode-musdb18 \
        mp4_root="${musdb18_mp4_root}" \
        wav_root="${musdb18_wav_root}" \
        subset="${subset}"

        # case 4: decode multiple .stem.mp4 files for all subsets
        subset="all"  # set subset to all

        audyn-decode-musdb18 \
        mp4_root="${musdb18_mp4_root}" \
        wav_root="${musdb18_wav_root}" \
        subset="${subset}"

    """
    decode_musdb18(config)


def decode_musdb18(config: DictConfig) -> None:
    frames_per_chunk = config.frames_per_chunk

    if config.mp4_root is None:
        assert config.wav_root is None
        assert config.mp4_path is not None

        mp4_path = config.mp4_path

        if config.wav_path is None:
            wav_path = os.path.join(mp4_path.replace(".stem.mp4", ""), "SOURCE.wav")
        else:
            wav_path = config.wav_path

        decode_file(mp4_path, wav_path, frames_per_chunk=frames_per_chunk)
    else:
        assert config.mp4_path is None
        assert config.wav_path is None

        mp4_root = config.mp4_root

        if config.wav_root is None:
            wav_root = mp4_root
        else:
            wav_root = config.wav_root

        subset = config.subset

        if subset is None:
            decode_folder(mp4_root, wav_root, frames_per_chunk=frames_per_chunk)
        else:
            if subset == "all":
                for _subset in ["train", "validation", "test"]:
                    mp4_dir = os.path.join(mp4_root, _subset)
                    wav_dir = os.path.join(wav_root, _subset)

                    if _subset == "validation":
                        mp4_dir = os.path.join(mp4_root, "train")
                        wav_dir = os.path.join(wav_root, "train")

                    decode_folder(
                        mp4_dir, wav_dir, subset=_subset, frames_per_chunk=frames_per_chunk
                    )
            else:
                mp4_dir = os.path.join(mp4_root, subset)
                wav_dir = os.path.join(wav_root, subset)

                if subset == "validation":
                    mp4_dir = os.path.join(mp4_root, "train")
                    wav_dir = os.path.join(wav_root, "train")

                decode_folder(mp4_dir, wav_dir, subset=subset, frames_per_chunk=frames_per_chunk)


def decode_folder(
    mp4_dir: str, wav_dir: str, subset: Optional[str] = None, frames_per_chunk: int = 4096
) -> None:
    """Decode .stem.mp4 files under and encode them as .wav files under wav_dir."""
    mp4_paths = sorted(glob.glob(os.path.join(mp4_dir, "*.stem.mp4")))
    track_names = []

    for mp4_path in mp4_paths:
        track_name = os.path.basename(mp4_path)
        track_name = track_name.replace(".stem.mp4", "")

        if subset is None:
            track_names.append(subset)
        else:
            if subset == "train":
                if track_name not in train_track_names:
                    continue
            elif subset == "validation":
                if track_name not in validation_track_names:
                    continue
            elif subset == "test":
                if track_name not in test_track_names:
                    continue
            else:
                raise ValueError(f"{subset} is not supported as subset.")

            track_names.append(track_name)

    if IS_TQDM_AVAILABLE:
        pbar = tqdm(track_names)
    else:
        pbar = track_names

    for track_name in pbar:
        mp4_path = os.path.join(mp4_dir, f"{track_name}.stem.mp4")
        wav_path = os.path.join(wav_dir, track_name, "SOURCE.wav")

        decode_file(mp4_path, wav_path, frames_per_chunk=frames_per_chunk)

    if IS_TQDM_AVAILABLE:
        pbar.close()


def decode_file(mp4_path: str, wav_path: str, frames_per_chunk: int = 44100) -> None:
    """Decode .stem.mp4 file and encode it as .wav files."""
    track_dir = os.path.dirname(wav_path)

    if track_dir:
        os.makedirs(track_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_mp4_path = os.path.join(temp_dir, "raw.stem.mp4")
        temp_wav_path = os.path.join(temp_dir, "SOURCE.wav")

        shutil.copy2(mp4_path, temp_mp4_path)

        for stream_idx, source in enumerate(["mixture"] + sources):
            reader = StreamReader(temp_mp4_path)
            writer = StreamWriter(temp_wav_path.replace("SOURCE", source))

            stream_info = reader.get_src_stream_info(stream_idx)
            sample_rate = stream_info.sample_rate
            num_channels = stream_info.num_channels

            assert sample_rate == 44100

            sample_rate = int(sample_rate)

            reader.add_audio_stream(
                frames_per_chunk=frames_per_chunk,
                stream_index=stream_idx,
            )
            writer.add_audio_stream(
                sample_rate=sample_rate,
                num_channels=num_channels,
            )

            with writer.open():
                for (chunk,) in reader.stream():
                    writer.write_audio_chunk(0, chunk)

            shutil.copy2(
                temp_wav_path.replace("SOURCE", source), wav_path.replace("SOURCE", source)
            )
