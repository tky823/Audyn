import torch

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
