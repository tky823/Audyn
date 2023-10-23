import pytest
import torch

from audyn.utils.data.postprocess import slice_feautures


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


def create_arange_batch(end: int, batch_size: int = 4) -> torch.Tensor:
    sequence = torch.arange(batch_size * end)
    batch_sequence = sequence.view(batch_size, end)

    return batch_sequence
