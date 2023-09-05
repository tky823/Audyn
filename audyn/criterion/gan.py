import torch.nn as nn


class GANCriterion:
    """Base class of criterion for GAN."""

    def __init__(self, generator: nn.Module, discriminator: nn.Module) -> None:
        self.generator = generator
        self.discriminator = discriminator
