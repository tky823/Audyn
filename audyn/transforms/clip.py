import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as vT
from packaging import version
from PIL.Image import Image

IS_TORCHVISION_LT_0_16 = version.parse(torchvision.__version__) < version.parse("0.16")


class OpenAICLIPImageTransform(nn.Module):
    def __init__(self, size: int = 224) -> None:
        super().__init__()

        transforms = [
            vT.Resize(size, interpolation=vT.InterpolationMode.BICUBIC),
            vT.CenterCrop(size),
        ]

        if IS_TORCHVISION_LT_0_16:
            transforms.append(vT.ToTensor())
            transforms.append(vT.ConvertImageDtype(torch.float32))
        else:
            transforms.append(vT.ToImage())
            transforms.append(vT.ToDtype(torch.float32, scale=True))

        transforms.append(
            vT.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            )
        )

        self.transforms = vT.Compose(transforms)

    def forward(self, input: torch.Tensor | Image) -> torch.Tensor:
        output = self.transforms(input)

        return output
