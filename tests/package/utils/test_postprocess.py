import math

import pytest
import torch

from audyn.utils.data.postprocess import slice_feautures, take_log_features


@pytest.mark.parametrize("random_slice", [True, False])
def test_slice_features(random_slice: bool) -> None:
    torch.manual_seed(0)

    batch_size = 4
    slice_length = 5
    max_fine_length = 4 * slice_length
    key_mapping = {
        "fine": "fine_slice",
        "coarse": "coarse_slice",
    }
    hop_lengths = {
        "fine": 1,
        "coarse": 2,
    }
    length_mapping = {
        "coarse": "coarse_length",
    }
    corse_length = torch.tensor(
        [
            max_fine_length // hop_lengths["coarse"],
            (max_fine_length - 2) // hop_lengths["coarse"],
            (max_fine_length - 4) // hop_lengths["coarse"],
            (max_fine_length - 8) // hop_lengths["coarse"],
        ],
        dtype=torch.long,
    )

    batch = {
        "fine": create_arange_batch(
            max_fine_length,
            batch_size=batch_size,
        ),
        "coarse": create_arange_batch(
            max_fine_length // hop_lengths["coarse"],
            batch_size=batch_size,
        ),
        "coarse_length": corse_length,
    }

    batch = slice_feautures(
        batch,
        slice_length,
        key_mapping=key_mapping,
        hop_lengths=hop_lengths,
        length_mapping=length_mapping,
        random_slice=random_slice,
    )

    assert torch.equal(batch["fine_slice"][:, 0], 2 * batch["coarse_slice"][:, 0])


@pytest.mark.parametrize("random_slice", [True, False])
def test_slice_features_length_dims(random_slice: bool) -> None:
    torch.manual_seed(0)

    batch_size = 4
    slice_length = 5
    max_fine_length = 4 * slice_length
    key_mapping = {
        "fine": "fine_slice",
        "coarse": "coarse_slice",
    }
    hop_lengths = {
        "fine": 1,
        "coarse": 2,
    }
    length_mapping = {
        "coarse": "coarse_length",
    }
    corse_length = torch.tensor(
        [
            max_fine_length // hop_lengths["coarse"],
            (max_fine_length - 2) // hop_lengths["coarse"],
            (max_fine_length - 4) // hop_lengths["coarse"],
            (max_fine_length - 8) // hop_lengths["coarse"],
        ],
        dtype=torch.long,
    )

    batch = {
        "fine": create_arange_batch(
            max_fine_length,
            batch_size=batch_size,
        ),
        "coarse": create_arange_batch(
            max_fine_length // hop_lengths["coarse"],
            batch_size=batch_size,
        ),
        "coarse_length": corse_length,
    }
    batch["coarse"] = batch["coarse"].unsqueeze(-1)
    length_dims = {
        "fine": -1,
        "coarse": -2,
    }

    batch = slice_feautures(
        batch,
        slice_length,
        key_mapping=key_mapping,
        hop_lengths=hop_lengths,
        length_mapping=length_mapping,
        length_dims=length_dims,
        random_slice=random_slice,
    )

    assert batch["fine_slice"].size(-1) == slice_length
    assert batch["coarse_slice"].size(-2) == math.ceil(slice_length / hop_lengths["coarse"])


def test_take_log_features() -> None:
    torch.manual_seed(0)

    batch_size = 4
    length = 10
    key_mapping = {
        "input": "log_input",
    }

    def add_flooring(input: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        return input + eps

    batch = {
        "input": torch.abs(
            torch.randint(
                0,
                2,
                (batch_size, length),
                dtype=torch.float,
            )
        ),
    }

    batch = take_log_features(
        batch,
        key_mapping=key_mapping,
        flooring_fn=add_flooring,
    )

    for value in batch.values():
        assert torch.isfinite(value).all()


def create_arange_batch(end: int, batch_size: int = 4) -> torch.Tensor:
    sequence = torch.arange(batch_size * end)
    batch_sequence = sequence.view(batch_size, end)

    return batch_sequence
