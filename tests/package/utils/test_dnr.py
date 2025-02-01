import os
import tempfile
import warnings
from typing import Dict, List

import torch
import torchaudio
from dummy import allclose

from audyn.utils.data.dnr import (
    sources,
    v2_test_track_names,
    v2_train_track_names,
    v2_validation_track_names,
)
from audyn.utils.data.dnr.dataset import DNR, Track


def test_dnr() -> None:
    sample_rate = 24000
    num_frames = 48000

    with tempfile.TemporaryDirectory() as temp_dir:
        _ = _save_dummy_dnr(root=temp_dir, sample_rate=sample_rate, num_frames=num_frames)

        # To ignore "Track is not found."
        warnings.simplefilter("ignore", UserWarning)
        dataset = DNR(temp_dir, subset="train")
        warnings.resetwarnings()

        for track in dataset:
            track: Track
            track.frame_offset = num_frames // 4
            track.num_frames = num_frames // 2

            speech, _ = track.speech
            music, _ = track.music
            effect, _ = track.effect
            mixture, _ = track.mixture
            stems, _ = track.stems

            assert track.name in v2_train_track_names
            allclose(speech + music + effect, mixture, atol=1e-4)
            allclose(stems[0], mixture, atol=1e-4)
            allclose(stems[1], speech)
            allclose(stems[2], music)
            allclose(stems[3], effect)

            break


def _save_dummy_dnr(root: str, sample_rate: int, num_frames: int) -> Dict[str, List[int]]:
    g = torch.Generator()
    g.manual_seed(0)

    num_channels = 2

    train_track_names = v2_train_track_names[:10]
    validation_track_names = v2_validation_track_names[:10]
    test_track_names = v2_test_track_names[:10]

    track_names = {
        "train": train_track_names,
        "validation": validation_track_names,
        "test": test_track_names,
    }

    for subset_name, track_names in zip(
        ["tr", "cv", "tt"], [train_track_names, validation_track_names, test_track_names]
    ):
        for track_name in track_names:
            track_dir = os.path.join(root, subset_name, f"{track_name}")

            os.makedirs(track_dir, exist_ok=True)

            mixture = 0

            for source in sources:
                if source in ["speech", "music"]:
                    _source = source
                elif source == "effect":
                    _source = "sfx"
                else:
                    raise ValueError(f"Invalid source {source} is given.")

                path = os.path.join(track_dir, f"{_source}.wav")
                waveform = torch.randn((num_channels, num_frames), generator=g)
                max_amplitude = torch.max(torch.abs(waveform))
                waveform = 0.1 * (waveform / max_amplitude)
                mixture = mixture + waveform
                torchaudio.save(path, waveform, sample_rate)

            path = os.path.join(track_dir, "mix.wav")
            torchaudio.save(path, mixture, sample_rate)

    return track_names
