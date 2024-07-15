import pytest
import torch

from audyn.criterion.hifigan import DiscriminatorHingeLoss, FeatureMatchingLoss, MSELoss
from audyn.models.hifigan import Generator, MultiPeriodDiscriminator


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_hifigan_mse_loss(reduction: str) -> None:
    batch_size = 4
    in_channels = 80
    length = 32

    generator = Generator.build_from_default_config(variation="v3")
    discriminator = MultiPeriodDiscriminator.build_from_default_config()

    input = torch.randn((batch_size, in_channels, length))
    generator_output = generator(input)
    prob, _ = discriminator(generator_output)

    target = 0
    criterion = MSELoss(target, reduction=reduction)
    loss = criterion(prob)

    assert loss.size() == ()


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_discriminator_hinge_loss(reduction: str) -> None:
    batch_size = 4
    in_channels = 80
    length = 32

    generator = Generator.build_from_default_config(variation="v3")
    discriminator = MultiPeriodDiscriminator.build_from_default_config()

    input = torch.randn((batch_size, in_channels, length))
    generator_output = generator(input)
    prob, _ = discriminator(generator_output)

    minimize = False
    criterion = DiscriminatorHingeLoss(minimize, reduction=reduction)
    loss = criterion(prob)

    assert loss.size() == ()


def test_feature_matching_loss() -> None:
    batch_size = 4
    in_channels = 80
    length = 32

    generator = Generator.build_from_default_config(variation="v3")
    discriminator = MultiPeriodDiscriminator.build_from_default_config()

    input = torch.randn((batch_size, in_channels, length))
    generator_output = generator(input)
    target = torch.randn_like(generator_output)
    _, feature_input = discriminator(generator_output)
    _, feature_target = discriminator(target)

    criterion = FeatureMatchingLoss()
    loss = criterion(feature_input, feature_target)

    assert loss.size() == ()
