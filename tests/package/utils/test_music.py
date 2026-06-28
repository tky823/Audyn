import math

import torch

from audyn.utils.music import (
    compute_bipartite_match_precision_recall_fscore,
    hz_to_midi,
    midi_to_hz,
    note_to_hz,
)


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


def test_bipartite_match() -> None:
    input = torch.tensor(
        [
            [0.6, 1.1, 1.6, 2.8, 3.5],
            [1.0, 2.0, 0.0, 0.0, 0.0],
            [1.1, 1.9, 3.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0, 0.0],
        ]
    )
    target = torch.tensor(
        [
            [1.0, 1.5, 2.2, 3.0],
            [1.0, 2.5, 0.0, 0.0],
            [1.0, 2.0, 3.0, 0.0],
            [5.0, 6.0, 7.0, 0.0],
        ]
    )
    input_lengths = torch.tensor([5, 2, 3, 4])
    target_lengths = torch.tensor([4, 2, 3, 3])
    expected_precision = torch.tensor([0.6, 1, 1, 0])
    expected_recall = torch.tensor([0.75, 1, 1, 0])
    expected_f_score = (
        2
        * (expected_precision * expected_recall)
        / torch.clamp(expected_precision + expected_recall, min=1)
    )

    precision, recall, f_score = compute_bipartite_match_precision_recall_fscore(
        input, target, input_lengths=input_lengths, target_lengths=target_lengths
    )

    assert torch.allclose(precision, expected_precision)
    assert torch.allclose(recall, expected_recall)
    assert torch.allclose(f_score, expected_f_score)
