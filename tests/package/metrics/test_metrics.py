import os
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dummy.utils import select_random_port, set_ddp_environment

from audyn.metrics import MeanMetric


def test_mean_metric() -> None:
    torch.manual_seed(0)

    num_samples = 10
    samples = torch.randn((num_samples,))
    expected_mean = torch.mean(samples)

    # torch.Tensor
    metric = MeanMetric()

    for sample in samples:
        metric.update(sample)

    mean = metric.compute()

    assert torch.allclose(mean, expected_mean)

    # list
    metric = MeanMetric()

    for sample in samples.tolist():
        metric.update(sample)

    mean = metric.compute()

    assert torch.allclose(mean, expected_mean)


def test_mean_metric_ddp() -> None:
    port = select_random_port()
    seed = 0
    world_size = 4

    batch_size = 8

    torch.manual_seed(seed)
    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = mp.get_context("spawn")

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = ctx.Process(
                target=run_mean_metric,
                args=(rank, world_size, port),
                kwargs={
                    "batch_size": batch_size,
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
        reference_loss = reference_state_dict["loss"]
        reference_metric = reference_state_dict["metric"]

        gathered_loss = [reference_loss]

        for rank in range(1, world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            state_dict = torch.load(path, map_location="cpu")
            loss = state_dict["loss"]
            metric = state_dict["metric"]

            assert torch.allclose(reference_metric, metric)

            gathered_loss.append(loss)

        gathered_loss = torch.cat(gathered_loss, dim=0)
        gathered_metric = torch.mean(gathered_loss)

        assert torch.allclose(reference_metric, gathered_metric)


def run_mean_metric(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    seed: int = 0,
    path: str = None,
) -> None:
    set_ddp_environment(rank, world_size, port)

    dist.init_process_group(backend="gloo", timeout=timedelta(minutes=1))
    torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(rank)

    metric = MeanMetric()

    loss = torch.randn((batch_size,), generator=g)

    for _loss in loss:
        metric.update(_loss)

    state_dict = {
        "loss": loss,
        "metric": metric.compute(),
    }
    torch.save(state_dict, path)

    dist.destroy_process_group()
