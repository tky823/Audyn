import os
import tempfile

import pytest
from audyn_test import allclose
from omegaconf import OmegaConf

from audyn.bin.decode_musdb18 import decode_musdb18
from audyn.bin.download_musdb18 import download_musdb18
from audyn.utils.audio import list_audio_backends
from audyn.utils.data.musdb18.dataset import MUSDB18, Track


@pytest.mark.slow
def test_download_musdb18_7s() -> None:
    if "ffmpeg" not in list_audio_backends():
        pytest.skip("FFmpeg is not supported.")

    with tempfile.TemporaryDirectory() as temp_dir:
        musdb18_dir = os.path.join(temp_dir, "MUSDB18-7s")
        subset = "test"

        config = OmegaConf.create(
            {
                "type": "7s",
                "root": temp_dir,
                "musdb18_root": musdb18_dir,
                "chunk_size": 16384,
                "unpack": None,
            }
        )
        download_musdb18(config)

        config = OmegaConf.create(
            {
                "mp4_root": musdb18_dir,
                "wav_root": musdb18_dir,
                "mp4_path": None,
                "wav_path": None,
                "subset": subset,
                "frames_per_chunk": 44100,
            }
        )
        decode_musdb18(config)

        wav_dataset = MUSDB18(musdb18_dir, subset=subset, ext="wav")
        mp4_dataset = MUSDB18(musdb18_dir, subset=subset, ext="mp4")

        for wav_track, mp4_track in zip(wav_dataset, mp4_dataset):
            wav_track: Track
            mp4_track: Track
            waveform_wav, _ = wav_track.stems
            waveform_mp4, _ = mp4_track.stems

            allclose(waveform_wav, waveform_mp4, atol=1e-4)
