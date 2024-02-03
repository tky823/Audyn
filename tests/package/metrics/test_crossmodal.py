import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dummy.utils import select_random_port

from audyn.metrics.crossmodal import (
    CrossModalEmbeddingMeanAveragePrecision,
    CrossModalEmbeddingMedianRank,
)

parameters_mink = [0, 1]


@pytest.mark.parametrize("mink", parameters_mink)
def test_crossmodal_mean_average_precision(mink: int) -> None:
    torch.manual_seed(0)

    k = 5
    batch_size = 4
    num_total_samples = 10
    embedding_dim = 3

    key = torch.randn((num_total_samples, embedding_dim))
    index = torch.arange(num_total_samples)

    expected_map_k = torch.tensor(1.0)

    # item-wise operation
    metric = CrossModalEmbeddingMeanAveragePrecision(k, mink=mink)

    for sample_idx, _key in enumerate(key):
        _query = _key  # i.e. oracle
        _index = index[sample_idx]
        metric.update(_query, key, index=_index)

    map_k = metric.compute()

    assert torch.allclose(map_k, expected_map_k)

    # batch-wise operation
    metric = CrossModalEmbeddingMeanAveragePrecision(k, mink=mink)

    for start_idx in range(0, num_total_samples, batch_size):
        _key = key[start_idx : start_idx + batch_size]
        _query = _key  # i.e. oracle
        _index = index[start_idx : start_idx + batch_size]
        metric.update(_query, key, index=_index)

    map_k = metric.compute()

    assert torch.allclose(map_k, expected_map_k)


@pytest.mark.parametrize("mink", parameters_mink)
@pytest.mark.parametrize("strategy", ["oracle", "random"])
def test_crossmodal_mean_average_precision_ddp_itemwise(mink: int, strategy: str) -> None:
    port = select_random_port()
    seed = 0
    world_size = 4

    k = 5
    batch_size = 5
    embedding_dim = 3

    torch.manual_seed(seed)
    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = mp.get_context("spawn")

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = ctx.Process(
                target=run_crossmodal_mean_average_precision_itemwise,
                args=(rank, world_size, port),
                kwargs={
                    "batch_size": batch_size,
                    "embedding_dim": embedding_dim,
                    "strategy": strategy,
                    "k": k,
                    "mink": mink,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        rank = 0
        path = os.path.join(temp_dir, f"{rank}.pth")
        reference_state_dict = torch.load(path, map_location="cpu")
        reference_query = reference_state_dict["query"]
        reference_key = reference_state_dict["key"]
        reference_index = reference_state_dict["index"]
        reference_map_k = reference_state_dict["map_k"]

        gathered_query = [reference_query]
        gathered_key = [reference_key]
        gathered_index = [reference_index]

        for rank in range(1, world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            state_dict = torch.load(path, map_location="cpu")
            query = state_dict["query"]
            key = state_dict["key"]
            index = state_dict["index"]
            map_k = state_dict["map_k"]

            assert torch.allclose(map_k, reference_map_k)

            gathered_query.append(query)
            gathered_key.append(key)
            gathered_index.append(index)

        gathered_query = torch.stack(gathered_query, dim=0)
        gathered_key = torch.stack(gathered_key, dim=0)
        gathered_index = torch.stack(gathered_index, dim=0)

        query = gathered_query.view(-1, embedding_dim)
        key = gathered_key.view(-1, embedding_dim)
        offset = torch.arange(0, world_size * batch_size, batch_size).unsqueeze(dim=-1)
        index = gathered_index + offset
        index = index.view(-1)

    if strategy == "oracle":
        expected_map_k = torch.tensor(1.0)
    else:
        expected_map_k = None

    # single device
    metric = CrossModalEmbeddingMeanAveragePrecision(k, mink=mink)

    for _query, _index in zip(query, index):
        metric.update(_query, key, index=_index)

    map_k = metric.compute()

    if expected_map_k is not None:
        assert torch.allclose(map_k, expected_map_k)

    assert torch.allclose(map_k, reference_map_k)


@pytest.mark.parametrize("mink", parameters_mink)
@pytest.mark.parametrize("strategy", ["oracle", "random"])
def test_crossmodal_mean_average_precision_ddp_batchwise(mink: int, strategy: str) -> None:
    port = select_random_port()
    seed = 0
    world_size = 4

    k = 5
    num_total_samples = 11
    batch_size = 5
    embedding_dim = 3

    torch.manual_seed(seed)
    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = mp.get_context("spawn")

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = ctx.Process(
                target=run_crossmodal_mean_average_precision_batchwise,
                args=(rank, world_size, port),
                kwargs={
                    "num_total_samples": num_total_samples,
                    "batch_size": batch_size,
                    "embedding_dim": embedding_dim,
                    "strategy": strategy,
                    "k": k,
                    "mink": mink,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        rank = 0
        path = os.path.join(temp_dir, f"{rank}.pth")
        reference_state_dict = torch.load(path, map_location="cpu")
        reference_query = reference_state_dict["query"]
        reference_key = reference_state_dict["key"]
        reference_index = reference_state_dict["index"]
        reference_map_k = reference_state_dict["map_k"]

        gathered_query = [reference_query]
        gathered_key = [reference_key]
        gathered_index = [reference_index]

        for rank in range(1, world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            state_dict = torch.load(path, map_location="cpu")
            query = state_dict["query"]
            key = state_dict["key"]
            index = state_dict["index"]
            map_k = state_dict["map_k"]

            assert torch.allclose(map_k, reference_map_k)

            gathered_query.append(query)
            gathered_key.append(key)
            gathered_index.append(index)

        gathered_query = torch.stack(gathered_query, dim=0)
        gathered_key = torch.stack(gathered_key, dim=0)
        gathered_index = torch.stack(gathered_index, dim=0)

        query = gathered_query.view(-1, embedding_dim)
        key = gathered_key.view(-1, embedding_dim)
        offset = torch.arange(0, world_size * num_total_samples, num_total_samples).unsqueeze(
            dim=-1
        )
        index = gathered_index + offset
        index = index.view(-1)

    if strategy == "oracle":
        expected_map_k = torch.tensor(1.0)
    else:
        expected_map_k = None

    # single device
    metric = CrossModalEmbeddingMeanAveragePrecision(k, mink=mink)

    for start_idx in range(0, world_size * num_total_samples, batch_size):
        _query = query[start_idx : start_idx + batch_size]
        _index = index[start_idx : start_idx + batch_size]
        metric.update(_query, key, index=_index)

    map_k = metric.compute()

    if expected_map_k is not None:
        assert torch.allclose(map_k, expected_map_k)

    assert torch.allclose(map_k, reference_map_k)


@pytest.mark.parametrize("mink", parameters_mink)
def test_crossmodal_median_rank(mink: int) -> None:
    torch.manual_seed(0)

    batch_size = 4
    num_total_samples = 10
    embedding_dim = 3

    key = torch.randn((num_total_samples, embedding_dim))
    index = torch.arange(num_total_samples)

    expected_medR = torch.full((), fill_value=mink)

    # item-wise operation
    metric = CrossModalEmbeddingMedianRank(mink=mink)

    for sample_idx, _key in enumerate(key):
        _query = _key  # i.e. oracle
        _index = index[sample_idx]
        metric.update(_query, key, index=_index)

    medR = metric.compute()

    assert torch.allclose(medR, expected_medR)

    # batch-wise operation
    metric = CrossModalEmbeddingMedianRank(mink=mink)

    for start_idx in range(0, num_total_samples, batch_size):
        _key = key[start_idx : start_idx + batch_size]
        _query = _key  # i.e. oracle
        _index = index[start_idx : start_idx + batch_size]
        metric.update(_query, key, index=_index)

    medR = metric.compute()

    assert torch.allclose(medR, expected_medR)


@pytest.mark.parametrize("mink", parameters_mink)
@pytest.mark.parametrize("strategy", ["oracle", "random"])
def test_crossmodal_median_rank_ddp_itemwise(mink: int, strategy: str) -> None:
    port = select_random_port()
    seed = 0
    world_size = 4

    batch_size = 5
    embedding_dim = 3

    torch.manual_seed(seed)
    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = mp.get_context("spawn")

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = ctx.Process(
                target=run_crossmodal_median_rank_itemwise,
                args=(rank, world_size, port),
                kwargs={
                    "batch_size": batch_size,
                    "embedding_dim": embedding_dim,
                    "strategy": strategy,
                    "mink": mink,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        rank = 0
        path = os.path.join(temp_dir, f"{rank}.pth")
        reference_state_dict = torch.load(path, map_location="cpu")
        reference_query = reference_state_dict["query"]
        reference_key = reference_state_dict["key"]
        reference_index = reference_state_dict["index"]
        reference_medR = reference_state_dict["medR"]

        gathered_query = [reference_query]
        gathered_key = [reference_key]
        gathered_index = [reference_index]

        for rank in range(1, world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            state_dict = torch.load(path, map_location="cpu")
            query = state_dict["query"]
            key = state_dict["key"]
            index = state_dict["index"]
            medR = state_dict["medR"]

            assert torch.allclose(medR, reference_medR)

            gathered_query.append(query)
            gathered_key.append(key)
            gathered_index.append(index)

        gathered_query = torch.stack(gathered_query, dim=0)
        gathered_key = torch.stack(gathered_key, dim=0)
        gathered_index = torch.stack(gathered_index, dim=0)

        query = gathered_query.view(-1, embedding_dim)
        key = gathered_key.view(-1, embedding_dim)
        offset = torch.arange(0, world_size * batch_size, batch_size).unsqueeze(dim=-1)
        index = gathered_index + offset
        index = index.view(-1)

    if strategy == "oracle":
        expected_medR = torch.full((), fill_value=mink)
    else:
        expected_medR = None

    # single device
    metric = CrossModalEmbeddingMedianRank(mink=mink)

    for _query, _index in zip(query, index):
        metric.update(_query, key, index=_index)

    medR = metric.compute()

    if expected_medR is not None:
        assert torch.allclose(medR, expected_medR)

    assert torch.allclose(medR, reference_medR)


@pytest.mark.parametrize("mink", parameters_mink)
@pytest.mark.parametrize("strategy", ["oracle", "random"])
def test_crossmodal_median_rank_ddp_batchwise(mink: int, strategy: str) -> None:
    port = select_random_port()
    seed = 0
    world_size = 4

    num_total_samples = 11
    batch_size = 5
    embedding_dim = 3

    torch.manual_seed(seed)
    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = mp.get_context("spawn")

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = ctx.Process(
                target=run_crossmodal_median_rank_batchwise,
                args=(rank, world_size, port),
                kwargs={
                    "num_total_samples": num_total_samples,
                    "batch_size": batch_size,
                    "embedding_dim": embedding_dim,
                    "strategy": strategy,
                    "mink": mink,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        rank = 0
        path = os.path.join(temp_dir, f"{rank}.pth")
        reference_state_dict = torch.load(path, map_location="cpu")
        reference_query = reference_state_dict["query"]
        reference_key = reference_state_dict["key"]
        reference_index = reference_state_dict["index"]
        reference_medR = reference_state_dict["medR"]

        gathered_query = [reference_query]
        gathered_key = [reference_key]
        gathered_index = [reference_index]

        for rank in range(1, world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            state_dict = torch.load(path, map_location="cpu")
            query = state_dict["query"]
            key = state_dict["key"]
            index = state_dict["index"]
            medR = state_dict["medR"]

            assert torch.allclose(medR, reference_medR)

            gathered_query.append(query)
            gathered_key.append(key)
            gathered_index.append(index)

        gathered_query = torch.stack(gathered_query, dim=0)
        gathered_key = torch.stack(gathered_key, dim=0)
        gathered_index = torch.stack(gathered_index, dim=0)

        query = gathered_query.view(-1, embedding_dim)
        key = gathered_key.view(-1, embedding_dim)
        offset = torch.arange(0, world_size * num_total_samples, num_total_samples).unsqueeze(
            dim=-1
        )
        index = gathered_index + offset
        index = index.view(-1)

    if strategy == "oracle":
        expected_medR = torch.full((), fill_value=mink)
    else:
        expected_medR = None

    # single device
    metric = CrossModalEmbeddingMedianRank(mink=mink)

    for start_idx in range(0, world_size * num_total_samples, batch_size):
        _query = query[start_idx : start_idx + batch_size]
        _index = index[start_idx : start_idx + batch_size]
        metric.update(_query, key, index=_index)

    medR = metric.compute()

    if expected_medR is not None:
        assert torch.allclose(medR, expected_medR)

    assert torch.allclose(medR, reference_medR)


def run_crossmodal_mean_average_precision_itemwise(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int = 5,
    embedding_dim: int = 3,
    strategy: str = "oracle",
    k: int = 5,
    mink: int = 0,
    seed: int = 0,
    path: str = None,
) -> None:
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    num_threads = torch.get_num_threads()
    num_threads = max(num_threads // world_size, 1)
    torch.set_num_threads(num_threads)

    dist.init_process_group(backend="gloo")
    torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(rank)

    metric = CrossModalEmbeddingMeanAveragePrecision(k, mink=mink)
    key = torch.randn((batch_size, embedding_dim), generator=g)

    if strategy == "oracle":
        query = key
        index = torch.arange(batch_size)
    elif strategy == "random":
        query = torch.randn((batch_size, embedding_dim), generator=g)
        index = torch.randperm(batch_size, generator=g)
    else:
        raise ValueError(f"{strategy} is not supported as strategy.")

    for sample_idx in range(batch_size):
        _query = query[sample_idx]
        _index = index[sample_idx]
        metric.update(_query, key, index=_index)

    state_dict = {
        "query": query,
        "key": key,
        "index": index,
        "map_k": metric.compute(),
    }
    torch.save(state_dict, path)

    dist.destroy_process_group()


def run_crossmodal_mean_average_precision_batchwise(
    rank: int,
    world_size: int,
    port: int,
    num_total_samples: int = 11,
    batch_size: int = 5,
    embedding_dim: int = 3,
    strategy: str = "oracle",
    k: int = 5,
    mink: int = 0,
    seed: int = 0,
    path: str = None,
) -> None:
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    num_threads = torch.get_num_threads()
    num_threads = max(num_threads // world_size, 1)
    torch.set_num_threads(num_threads)

    dist.init_process_group(backend="gloo")
    torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(rank)

    metric = CrossModalEmbeddingMeanAveragePrecision(k, mink=mink)
    key = torch.randn((num_total_samples, embedding_dim), generator=g)

    if strategy == "oracle":
        query = key
        index = torch.arange(num_total_samples)
    elif strategy == "random":
        query = torch.randn((num_total_samples, embedding_dim), generator=g)
        index = torch.randperm(num_total_samples, generator=g)
    else:
        raise ValueError(f"{strategy} is not supported as strategy.")

    for start_idx in range(0, num_total_samples, batch_size):
        _query = query[start_idx : start_idx + batch_size]
        _index = index[start_idx : start_idx + batch_size]
        metric.update(_query, key, index=_index)

    state_dict = {
        "query": query,
        "key": key,
        "index": index,
        "map_k": metric.compute(),
    }
    torch.save(state_dict, path)

    dist.destroy_process_group()


def run_crossmodal_median_rank_itemwise(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int = 5,
    embedding_dim: int = 3,
    strategy: str = "oracle",
    mink: int = 0,
    seed: int = 0,
    path: str = None,
) -> None:
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    num_threads = torch.get_num_threads()
    num_threads = max(num_threads // world_size, 1)
    torch.set_num_threads(num_threads)

    dist.init_process_group(backend="gloo")
    torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(rank)

    metric = CrossModalEmbeddingMedianRank(mink=mink)
    key = torch.randn((batch_size, embedding_dim), generator=g)

    if strategy == "oracle":
        query = key
        index = torch.arange(batch_size)
    elif strategy == "random":
        query = torch.randn((batch_size, embedding_dim), generator=g)
        index = torch.randperm(batch_size, generator=g)
    else:
        raise ValueError(f"{strategy} is not supported as strategy.")

    for sample_idx in range(batch_size):
        _query = query[sample_idx]
        _index = index[sample_idx]
        metric.update(_query, key, index=_index)

    state_dict = {
        "query": query,
        "key": key,
        "index": index,
        "medR": metric.compute(),
    }
    torch.save(state_dict, path)

    dist.destroy_process_group()


def run_crossmodal_median_rank_batchwise(
    rank: int,
    world_size: int,
    port: int,
    num_total_samples: int = 11,
    batch_size: int = 5,
    embedding_dim: int = 3,
    strategy: str = "oracle",
    mink: int = 0,
    seed: int = 0,
    path: str = None,
) -> None:
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    num_threads = torch.get_num_threads()
    num_threads = max(num_threads // world_size, 1)
    torch.set_num_threads(num_threads)

    dist.init_process_group(backend="gloo")
    torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(rank)

    metric = CrossModalEmbeddingMedianRank(mink=mink)
    key = torch.randn((num_total_samples, embedding_dim), generator=g)

    if strategy == "oracle":
        query = key
        index = torch.arange(num_total_samples)
    elif strategy == "random":
        query = torch.randn((num_total_samples, embedding_dim), generator=g)
        index = torch.randperm(num_total_samples, generator=g)
    else:
        raise ValueError(f"{strategy} is not supported as strategy.")

    for start_idx in range(0, num_total_samples, batch_size):
        _query = query[start_idx : start_idx + batch_size]
        _index = index[start_idx : start_idx + batch_size]
        metric.update(_query, key, index=_index)

    state_dict = {
        "query": query,
        "key": key,
        "index": index,
        "medR": metric.compute(),
    }
    torch.save(state_dict, path)

    dist.destroy_process_group()
