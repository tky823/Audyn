import torch

from audyn.metrics.retrieval import MeanAveragePrecision


def test_mean_average_precision() -> None:
    torch.manual_seed(0)

    num_queries = 10
    max_items = 20
    k, mink = 5, 0

    metric = MeanAveragePrecision(k, mink=mink)

    num_ranks = torch.randint(1, max_items, (num_queries,), dtype=torch.long)
    num_ranks = num_ranks.tolist()
    ranks = []

    # oracle recommendation
    for _num_ranks in num_ranks:
        ranks.append(list(range(_num_ranks)))

    for rank in ranks:
        metric.update(rank)

    map_k = metric.compute()
    expected_map_k = torch.tensor(1.0)

    assert torch.allclose(map_k, expected_map_k)

    # examples
    k, mink = 3, 1
    ranks = [[1, 2], [2], [4, 2, 1, 3]]

    metric = MeanAveragePrecision(k, mink=mink)

    for rank in ranks:
        metric.update(rank, enforce_sorted=True)

    map_k = metric.compute()
    expected_map_k = torch.tensor(2.5 / 3)

    assert torch.allclose(map_k, expected_map_k)

    # TODO: add more tests
