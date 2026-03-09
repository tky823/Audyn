import torch
import torch.nn as nn
import torchvision.transforms.v2 as vT
from PIL.Image import Image


class OpenAICLIPImageTransform(nn.Module):
    def __init__(self, size: int = 224) -> None:
        super().__init__()

        transforms = [
            vT.Resize(size, interpolation=vT.InterpolationMode.BICUBIC),
            vT.CenterCrop(size),
            vT.ToImage(),
            vT.ToDtype(torch.float32, scale=True),
            vT.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ]
        self.transforms = vT.Compose(transforms)

    def forward(self, input: torch.Tensor | Image) -> torch.Tensor:
        output = self.transforms(input)

        return output
