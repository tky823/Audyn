import os
import tempfile
from typing import Tuple, Union

import torch
import torchaudio

__all__ = [
    "supported_audio_extensions",
    "decode_audio",
]

supported_audio_extensions = ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"]


def decode_audio(
    audio: bytes,
    ext: str,
    decode_audio_as_monoral: bool = True,
    decode_audio_as_waveform: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, f"audio.{ext}")

        with open(path, "wb") as f:
            f.write(audio)

        waveform, sample_rate = torchaudio.load(path)

    if decode_audio_as_monoral:
        waveform = waveform.mean(dim=0)

    if decode_audio_as_waveform:
        decoded = waveform
    else:
        decoded = waveform, sample_rate

    return decoded
