from typing import List, Union

__all__ = [
    "note_to_hz",
]


def note_to_hz(note: Union[str, List[str]]) -> Union[float, List[float]]:
    """Convert note to frequency.

    Args:
        note (str or list): Note or sequence of note.

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
    if isinstance(note, list):
        return [note_to_hz(_note) for _note in note]

    assert len(note) in [2, 3], "Invalid format is given as note."

    freq_a0 = 27.5
    octave = 12

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
