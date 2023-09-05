import torch.nn as nn


class BaseGAN(nn.Module):
    """Base class of GAN."""

    def __init__(self, generator: nn.Module, discriminator: nn.Module) -> None:
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
