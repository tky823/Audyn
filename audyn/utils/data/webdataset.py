import os
import tempfile
from typing import Tuple, Union

import torch
import torchaudio

__all__ = [
    "supported_class_extensions",
    "supported_text_extensions",
    "supported_torchdump_extensions",
    "supported_json_extensions",
    "supported_audio_extensions",
    "decode_audio",
]

supported_class_extensions = [
    "cls",
    "cls2",
    "class",
    "count",
    "index",
    "inx",
    "id",
]
supported_text_extensions = [
    "txt",
    "text",
    "transcript",
]
supported_torchdump_extensions = [
    "pth",
    # TODO: "pt"
]
supported_json_extensions = [
    "json",
    "jsn",
]
supported_audio_extensions = [
    "flac",
    "mp3",
    "sox",
    "wav",
    "m4a",
    "ogg",
    "wma",
]


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
