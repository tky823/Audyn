import torch

from audyn.metrics.retrieval import MeanAveragePrecision, MedianRank


def test_mean_average_precision() -> None:
    torch.manual_seed(0)

    num_queries = 10
    max_items = 20
    k, mink = 5, 0

    num_ranks = torch.randint(1, max_items, (num_queries,), dtype=torch.long)
    num_ranks = num_ranks.tolist()
    ranks = []

    # oracle recommendation
    for _num_ranks in num_ranks:
        ranks.append(list(range(_num_ranks)))

    metric = MeanAveragePrecision(k, mink=mink)

    for rank in ranks:
        metric.update(rank)

    map_k = metric.compute()
    expected_map_k = torch.tensor(1.0)

    assert torch.allclose(map_k, expected_map_k)

    # examples
    k, mink = 3, 0
    ranks = [[0, 1], [1], [3, 1, 0, 2]]

    metric = MeanAveragePrecision(k, mink=mink)

    for rank in ranks:
        metric.update(rank, enforce_sorted=True)

    map_k = metric.compute()
    expected_map_k = torch.tensor(2.5 / 3)

    assert torch.allclose(map_k, expected_map_k)

    mink = 1
    ranks = [[1, 2], [2], [4, 2, 1, 3]]

    metric = MeanAveragePrecision(k, mink=mink)

    for rank in ranks:
        metric.update(rank, enforce_sorted=True)

    map_k = metric.compute()
    expected_map_k = torch.tensor(2.5 / 3)

    assert torch.allclose(map_k, expected_map_k)


def test_median_rank() -> None:
    torch.manual_seed(0)

    num_queries = 10
    mink = 0

    # oracle recommendations
    ranks = torch.full((num_queries,), fill_value=mink, dtype=torch.long)
    expected_medR = torch.tensor(mink, dtype=torch.long)
    metric = MedianRank()

    for rank in ranks:
        metric.update(rank)

    medR = metric.compute()

    assert torch.allclose(medR, expected_medR)

    metric = MedianRank()

    for rank in ranks.tolist():
        metric.update(rank)

    medR = metric.compute()

    assert torch.allclose(medR, expected_medR)

    mink = 1
    ranks = torch.full((num_queries,), fill_value=mink, dtype=torch.long)
    expected_medR = torch.tensor(mink, dtype=torch.long)
    metric = MedianRank()

    for rank in ranks:
        metric.update(rank)

    medR = metric.compute()

    assert torch.allclose(medR, expected_medR)

    metric = MedianRank()

    for rank in ranks.tolist():
        metric.update(rank)

    medR = metric.compute()

    assert torch.allclose(medR, expected_medR)

    # examples
    mink = 0
    ranks = torch.tensor([0, 4, 6, 3])

    metric = MedianRank()

    for rank in ranks:
        metric.update(rank)

    medR = metric.compute()
    expected_medR = torch.median(ranks)

    assert torch.allclose(medR, expected_medR)

    metric = MedianRank()

    for rank in ranks.tolist():
        metric.update(rank)

    medR = metric.compute()
    expected_medR = torch.median(ranks)

    assert torch.allclose(medR, expected_medR)

    mink = 1
    ranks = torch.tensor([1, 5, 7, 4])

    metric = MedianRank()

    for rank in ranks:
        metric.update(rank)

    medR = metric.compute()
    expected_medR = torch.median(ranks)

    assert torch.allclose(medR, expected_medR)

    metric = MedianRank()

    for rank in ranks.tolist():
        metric.update(rank)

    medR = metric.compute()
    expected_medR = torch.median(ranks)

    assert torch.allclose(medR, expected_medR)
