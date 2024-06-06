from numbers import Number
from typing import List, Union

import torch

__all__ = [
    "note_to_hz",
    "midi_to_hz",
]


def note_to_hz(note: Union[str, List[str]]) -> Union[float, List[float]]:
    """Convert note to frequency.

    Args:
        note: Note or sequence of note.

    Returns:
        str or list: Frequency or sequence of frequency.

    Examples:

        >>> from audyn.utils.music import note_to_hz
        >>> note_to_hz("A4")
        440.0  # 440Hz
        >>> note_to_hz(["G5", "E6", "Bb5"])
        [783.9908719634984, 1318.5102276514795, 932.3275230361796]

    .. note::

        We assume ``A0`` is ``27.5`` Hz, i.e. ``A4`` is assumed to be 440Hz.

    """
    octave = 12
    freq_a0 = 27.5

    if isinstance(note, list):
        return [note_to_hz(_note) for _note in note]

    assert len(note) in [2, 3], "Invalid format is given as note."

    offset_mapping = {
        "Cb": -1,
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "E#": 5,
        "Fb": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
        "B#": 12,
    }

    note_pitch = int(note[-1])
    note_idx = octave * note_pitch + offset_mapping[note[:-1]] - offset_mapping["A"]

    freq = freq_a0 * 2 ** (note_idx / octave)

    return freq


def midi_to_hz(
    midi: Union[Number, List[Number], torch.Tensor]
) -> Union[float, List[float], torch.Tensor]:
    """Convert MIDI number to frequency.

    Args:
        midi: MIDI number or sequence of MIDI number. Numeric (``float`` and ``int``),
            and ``torch.Tensor`` are expected types.

    Returns:
        float, list, or torch.Tensor: Frequency or sequence of frequency.

    Examples:

        >>> import torch
        >>> from audyn.utils.music import midi_to_hz
        >>> midi = 69  # MIDI number of A4
        >>> midi_to_hz(69)
        440.0  # 440Hz
        >>> midi_to_hz([69, 70, 81])
        [440.0, 466.1637615180898, 880.0]
        >>> midi_to_hz(torch.tensor([69, 70, 81]))
        tensor([440.0000, 466.1638, 880.0000])

    .. note::

        We assume ``A0`` is ``27.5`` Hz, i.e. ``A4`` is assumed to be 440Hz.

    """
    octave = 12
    freq_a0 = 27.5

    if isinstance(midi, list):
        return [midi_to_hz(_midi) for _midi in midi]

    # 69: MIDI number of A0
    #  4: Number of octaves between A4 and A0
    note_idx = midi - 69 + 4 * octave
    freq = freq_a0 * 2 ** (note_idx / octave)

    return freq
