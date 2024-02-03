import copy
import os
import tempfile
from typing import List, Union

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dummy.utils import select_random_port

from audyn.metrics.retrieval import MeanAveragePrecision, MedianRank

parameters_mink = [0, 1]


@pytest.mark.parametrize("mink", parameters_mink)
def test_mean_average_precision_oracle(mink: int) -> None:
    torch.manual_seed(0)

    num_queries = 10
    max_items = 20
    k = 5

    num_ranks = torch.randint(1, max_items, (num_queries,), dtype=torch.long)
    num_ranks = num_ranks.tolist()
    ranks = []

    # oracle recommendation
    for _num_ranks in num_ranks:
        ranks.append(list(range(mink, _num_ranks + mink)))

    metric = MeanAveragePrecision(k, mink=mink)

    for rank in ranks:
        metric.update(rank)

    map_k = metric.compute()
    expected_map_k = torch.tensor(1.0)

    assert torch.allclose(map_k, expected_map_k)


@pytest.mark.parametrize("mink", parameters_mink)
def test_mean_average_precision_known_map(mink: int) -> None:
    torch.manual_seed(0)

    k = 3
    ranks = [[0 + mink, 1 + mink], [1 + mink], [3 + mink, 1 + mink, 0 + mink, 2 + mink]]

    metric = MeanAveragePrecision(k, mink=mink)

    for rank in ranks:
        metric.update(rank, enforce_sorted=True)

    map_k = metric.compute()
    expected_map_k = torch.tensor(2.5 / 3)

    assert torch.allclose(map_k, expected_map_k)


@pytest.mark.parametrize("mink", parameters_mink)
def test_mean_average_precision_ddp_oracle(mink: int) -> None:
    port = select_random_port()
    seed = 0
    world_size = 3

    torch.manual_seed(seed)

    num_queries = 10
    k = 5

    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for process_rank in range(world_size):
            path = os.path.join(temp_dir, f"{process_rank}.pth")
            process = mp.Process(
                target=run_mean_average_precision,
                args=(process_rank, world_size, port),
                kwargs={
                    "ranks": "random",
                    "k": k,
                    "mink": mink,
                    "num_queries": num_queries,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        process_rank = 0
        path = os.path.join(temp_dir, f"{process_rank}.pth")
        reference_state_dict = torch.load(path, map_location="cpu")
        reference_ranks = reference_state_dict["ranks"]
        reference_map_k = reference_state_dict["map_k"]

        gathered_ranks = copy.deepcopy(reference_ranks)

        for process_rank in range(1, world_size):
            path = os.path.join(temp_dir, f"{process_rank}.pth")
            state_dict = torch.load(path, map_location="cpu")
            ranks = state_dict["ranks"]
            map_k = state_dict["map_k"]

            assert torch.allclose(reference_map_k, map_k)

            gathered_ranks.extend(ranks)

    # confirm usage of DDP with single device for oracle retrieval
    metric = MeanAveragePrecision(k, mink=mink)

    for rank in gathered_ranks:
        metric.update(rank, enforce_sorted=True)

    gathered_map_k = metric.compute()
    expected_map_k = torch.tensor(1.0)

    assert torch.allclose(reference_map_k, gathered_map_k)
    assert torch.allclose(map_k, expected_map_k)


@pytest.mark.parametrize("mink", parameters_mink)
def test_mean_average_precision_ddp_know_map(mink: int) -> None:
    port = select_random_port()
    seed = 0
    world_size = 3

    torch.manual_seed(seed)

    k = 3
    ranks = [[0 + mink, 1 + mink], [1 + mink], [3 + mink, 1 + mink, 0 + mink, 2 + mink]]
    expected_map_k = torch.tensor(2.5 / 3)

    assert len(ranks) == world_size

    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for process_rank in range(world_size):
            path = os.path.join(temp_dir, f"{process_rank}.pth")
            process = mp.Process(
                target=run_mean_average_precision,
                args=(process_rank, world_size, port),
                kwargs={
                    "ranks": ranks[process_rank : process_rank + 1],
                    "k": k,
                    "mink": mink,
                    "num_queries": None,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        process_rank = 0
        path = os.path.join(temp_dir, f"{process_rank}.pth")
        reference_state_dict = torch.load(path, map_location="cpu")
        reference_ranks = reference_state_dict["ranks"]
        reference_map_k = reference_state_dict["map_k"]

        gathered_ranks: List[int] = copy.deepcopy(reference_ranks)

        for process_rank in range(1, world_size):
            path = os.path.join(temp_dir, f"{process_rank}.pth")
            state_dict = torch.load(path, map_location="cpu")
            ranks = state_dict["ranks"]
            map_k = state_dict["map_k"]

            assert torch.allclose(reference_map_k, map_k)

            gathered_ranks.extend(copy.deepcopy(ranks))

    # confirm usage of DDP with single device for oracle retrieval
    metric = MeanAveragePrecision(k, mink=mink)

    for rank in gathered_ranks:
        metric.update(rank, enforce_sorted=True)

    gathered_map_k = metric.compute()

    assert torch.allclose(reference_map_k, gathered_map_k)
    assert torch.allclose(gathered_map_k, expected_map_k)


@pytest.mark.parametrize("mink", parameters_mink)
def test_median_rank(mink: int) -> None:
    torch.manual_seed(0)

    num_queries = 10

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

    # examples
    ranks = torch.randint(mink, 5, (4,))

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


@pytest.mark.parametrize("ranks", ["oracle", "random"])
@pytest.mark.parametrize("mink", parameters_mink)
def test_median_rank_ddp(ranks: str, mink: int) -> None:
    port = select_random_port()
    seed = 0
    world_size = 4

    num_queries = 10

    if ranks == "oracle":
        expected_medR = torch.tensor(mink, dtype=torch.long)
    else:
        expected_medR = None

    torch.manual_seed(seed)
    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for process_rank in range(world_size):
            path = os.path.join(temp_dir, f"{process_rank}.pth")
            process = mp.Process(
                target=run_median_rank,
                args=(process_rank, world_size, port),
                kwargs={
                    "ranks": ranks,
                    "mink": mink,
                    "num_queries": num_queries,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        process_rank = 0
        path = os.path.join(temp_dir, f"{process_rank}.pth")
        reference_state_dict = torch.load(path, map_location="cpu")
        reference_ranks = reference_state_dict["ranks"]
        reference_medR = reference_state_dict["medR"]

        gathered_ranks = [reference_ranks]

        for process_rank in range(1, world_size):
            path = os.path.join(temp_dir, f"{process_rank}.pth")
            state_dict = torch.load(path, map_location="cpu")
            ranks = state_dict["ranks"]
            medR = state_dict["medR"]

            assert torch.allclose(reference_medR, medR)

            gathered_ranks.append(ranks)

        gathered_ranks = torch.cat(gathered_ranks, dim=0)

    # confirm usage of DDP with single device
    metric = MedianRank()

    for rank in gathered_ranks:
        metric.update(rank)

    gathered_medR = metric.compute()

    assert torch.allclose(reference_medR, gathered_medR)

    if expected_medR is not None:
        assert torch.allclose(gathered_medR, expected_medR)


def run_mean_average_precision(
    process_rank: int,
    world_size: int,
    port: int,
    ranks: Union[str, List[int]],
    k: int,
    mink: int = 0,
    num_queries: int = None,
    seed: int = 0,
    path: str = None,
) -> None:
    os.environ["LOCAL_RANK"] = str(process_rank)
    os.environ["RANK"] = str(process_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    num_threads = torch.get_num_threads()
    num_threads = max(num_threads // world_size, 1)
    torch.set_num_threads(num_threads)

    dist.init_process_group(backend="gloo")
    torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(process_rank)

    if isinstance(ranks, str):
        # use larger number as max_items for test
        max_items = k + 2
        num_ranks = torch.randint(1, max_items, (num_queries,), dtype=torch.long)
        num_ranks = num_ranks.tolist()
        ranks = []

        # oracle recommendation
        for _num_ranks in num_ranks:
            ranks.append(list(range(mink, _num_ranks + mink)))
    elif isinstance(ranks, list):
        # use given ranks
        assert num_queries is None
    else:
        raise ValueError(f"{type(ranks)} is not supported.")

    metric = MeanAveragePrecision(k, mink=mink)

    for rank in ranks:
        metric.update(rank, enforce_sorted=True)

    state_dict = {
        "ranks": ranks,
        "map_k": metric.compute(),
    }
    torch.save(state_dict, path)

    dist.destroy_process_group()


def run_median_rank(
    process_rank: int,
    world_size: int,
    port: int,
    ranks: str,
    mink: int = 0,
    num_queries: int = None,
    seed: int = 0,
    path: str = None,
) -> None:
    os.environ["LOCAL_RANK"] = str(process_rank)
    os.environ["RANK"] = str(process_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    num_threads = torch.get_num_threads()
    num_threads = max(num_threads // world_size, 1)
    torch.set_num_threads(num_threads)

    dist.init_process_group(backend="gloo")
    torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(process_rank)

    metric = MedianRank()

    if ranks == "oracle":
        ranks = torch.full((num_queries,), fill_value=mink, dtype=torch.long)
    elif ranks == "random":
        ranks = torch.randint(mink, num_queries, (num_queries,), dtype=torch.long)
    else:
        raise ValueError(f"{ranks} is not supported.")

    for rank in ranks:
        metric.update(rank)

    state_dict = {
        "ranks": ranks,
        "medR": metric.compute(),
    }
    torch.save(state_dict, path)

    dist.destroy_process_group()
