import glob
import os
import shutil
import tempfile

from omegaconf import DictConfig
from torchaudio.io import StreamReader, StreamWriter

from ..utils.data.musdb18 import sources
from ..utils.hydra import main as audyn_main


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
    chunk_size = config.chunk_size

    if config.mp4_root is None:
        assert config.wav_root is None
        assert config.mp4_path is not None

        mp4_path = config.mp4_path

        if config.wav_path is None:
            wav_path = os.path.join(mp4_path.replace(".stem.mp4", ""), "SOURCE.wav")
        else:
            wav_path = config.wav_path

        decode_file(mp4_path, wav_path, chunk_size=chunk_size)
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
            decode_folder(mp4_root, wav_root, chunk_size=chunk_size)
        else:
            mp4_dir = os.path.join(mp4_root, subset)
            wav_dir = os.path.join(wav_root, subset)

            decode_folder(mp4_dir, wav_dir, chunk_size=chunk_size)


def decode_folder(mp4_dir: str, wav_dir: str, chunk_size: int = 4096) -> None:
    """Decode .stem.mp4 files under and encode them as .wav files under wav_dir."""
    mp4_paths = sorted(glob.glob(os.path.join(mp4_dir, "*")))

    for mp4_path in mp4_paths:
        track_name = os.path.basename(mp4_path)
        wav_path = os.path.join(wav_dir, track_name, "SOURCE.wav")

        decode_file(mp4_path, wav_path, chunk_size=chunk_size)


def decode_file(mp4_path: str, wav_path: str, chunk_size: int = 4096) -> None:
    """Decode .stem.mp4 file and encode it as .wav files."""
    track_dir = os.path.dirname(wav_path)

    if track_dir:
        os.makedirs(track_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_mp4_path = os.path.join(temp_dir, "raw.stem.mp4")
        temp_wav_path = os.path.join(temp_dir, "SOURCE.mp4")

        shutil.copy2(mp4_path, temp_mp4_path)

        for stream_idx, source in enumerate(["mixture"] + sources):
            reader = StreamReader(temp_mp4_path)
            writer = StreamWriter(temp_wav_path.replace("SOURCE", source))

            stream_info = reader.get_src_stream_info(stream_idx)
            sample_rate = stream_info.sample_rate
            num_channels = stream_info.num_channels

            assert sample_rate == 44100

            sample_rate = int(sample_rate)

            reader.add_basic_audio_stream(
                frames_per_chunk=chunk_size,
                stream_index=stream_idx,
                format=None,
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
