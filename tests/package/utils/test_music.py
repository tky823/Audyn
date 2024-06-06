import math

import torch

from audyn.utils.music import hz_to_midi, midi_to_hz, note_to_hz


def test_note_to_hz() -> None:
    assert math.isclose(note_to_hz("A4"), 440)

    for hz, expected_hz in zip(
        note_to_hz(["G5", "E6", "Bb5"]),
        [
            783.9908719634984,
            1318.5102276514795,
            932.3275230361796,
        ],
    ):
        assert math.isclose(hz, expected_hz)


def test_midi_to_hz() -> None:
    assert math.isclose(midi_to_hz(69), 440)

    for hz, expected_hz in zip(
        midi_to_hz([69, 70, 81]),
        [440.0, 466.1637615180898, 880.0],
    ):
        assert math.isclose(hz, expected_hz)

    assert torch.allclose(
        midi_to_hz(torch.tensor([69, 70, 81])), torch.tensor([440.0, 466.1637615180898, 880.0])
    )


def test_hz_to_midi() -> None:
    assert math.isclose(hz_to_midi(440), 69)

    for midi, expected_midi in zip(
        hz_to_midi([440.0, 466.1637615180898, 880.0]),
        [69, 70, 81],
    ):
        assert math.isclose(midi, expected_midi)

    assert torch.allclose(
        hz_to_midi(torch.tensor([440.0, 466.1637615180898, 880.0])),
        torch.tensor([69.0, 70.0, 81.0]),
    )
