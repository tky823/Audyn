import os
from collections import OrderedDict
from typing import List, Union

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.common_types import _size_1_t

from ..modules.sincnet import ConvBlock1d, QuadratureSincConvBlock1d
from ..utils._github import download_file_from_github_release


class PyannoteCompatibleSpeakerSegmenter(nn.Module):
    """Speaker segmentation model compatible with pyannote.audio PyanNet.

    The model encodes a waveform, processes frame-level features with a
    recurrent backbone, and predicts a score for each output class.
    """

    def __init__(
        self, encoder: Union["SincNet", nn.Module], backbone: nn.Module, classifier: nn.Module
    ) -> None:
        """Initialize the segmenter from its component modules.

        Args:
            encoder: Module that converts waveforms to frame-level features.
            backbone: Module that processes the encoded feature sequence.
            classifier: Module that maps each frame feature to class scores.

        """
        super().__init__()

        self.encoder = encoder
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute frame-level class scores.

        Args:
            input: Waveform tensor with shape (batch_size, channels, samples).

        Returns:
            torch.Tensor: Frame-level class scores of shape
                (batch_size, num_classes, num_frames).

        """
        x = self.encoder(input)
        x = self.backbone(x)
        output = self.classifier(x)

        return output

    @classmethod
    def build_from_config(
        cls, pretrained_model_name_or_path: str
    ) -> "PyannoteCompatibleSpeakerSegmenter":
        """Build a supported pyannote-compatible model architecture.

        Args:
            pretrained_model_name_or_path: Identifier of the supported model
                configuration. Currently, ``"pyannote-3.0"`` is supported.

        """
        if pretrained_model_name_or_path == "pyannote-3.0":
            # encoder
            num_encoder_features = [1, 80, 60, 60]
            encoder_kernel_size = [251, 5, 5]
            encoder_pool_size = [3, 3, 3]
            encoder_stride = 10

            # backbone
            in_channels = num_encoder_features[-1]
            out_channels = 128
            num_rnn_layers = 4
            num_proj_layers = 2
            dropout = 0.5
            is_causal = False

            # classifier
            num_classes = 7  # power set (0, A, B, C, A+B, A+C, B+C)

            encoder = SincNet(
                num_encoder_features,
                kernel_size=encoder_kernel_size,
                pool_size=encoder_pool_size,
                stride=encoder_stride,
            )
            backbone = SpeakerSegmenterBackbone(
                in_channels,
                out_channels,
                num_rnn_layers=num_rnn_layers,
                num_proj_layers=num_proj_layers,
                dropout=dropout,
                is_causal=is_causal,
            )
            classifier = nn.Conv1d(
                out_channels,
                num_classes,
                kernel_size=1,
                stride=1,
            )

            model = cls(encoder=encoder, backbone=backbone, classifier=classifier)
        else:
            raise ValueError(
                f"Unknown pretrained model {pretrained_model_name_or_path} is found. "
                "Supported model is 'pyannote-3.0'."
            )

        return model

    @classmethod
    def build_from_pretrained(
        cls, pretrained_model_name_or_path: str
    ) -> "PyannoteCompatibleSpeakerSegmenter":
        """Build a segmenter and load its pretrained parameters.

        Args:
            pretrained_model_name_or_path: Path to a checkpoint file, or the
                identifier of a supported pretrained model. Currently,
                ``"pyannote-3.0"`` is supported. When given the identifier,
                the checkpoint is downloaded to the model cache if needed.

        """
        from ..utils import instantiate, model_cache_dir

        pretrained_model_configs = {
            "pyannote-3.0": {
                "url": "https://github.com/tky823/Audyn/releases/download/v0.3.1/pyannote-3.0_speaker-segmenter.pth",  # noqa: E501
                "path": os.path.join(
                    model_cache_dir, "speaker-segmenter", "5c3bd42e", "pyannote-3.0.pth"
                ),
                "sha256": "5c3bd42ed55427a826362739ed6dfe41aa71503208e68e860edf9230342174d3",
            },
        }

        if os.path.exists(pretrained_model_name_or_path):
            state_dict = torch.load(
                pretrained_model_name_or_path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
            model_state_dict: OrderedDict = state_dict["model"]
            resolved_config = OmegaConf.create(state_dict["resolved_config"])
            pretrained_model_config = resolved_config.model
            model: PyannoteCompatibleSpeakerSegmenter = instantiate(pretrained_model_config)
            model.load_state_dict(model_state_dict)

            return model
        elif pretrained_model_name_or_path in pretrained_model_configs:
            config = pretrained_model_configs[pretrained_model_name_or_path]
            url = config["url"]
            path = config["path"]
            download_file_from_github_release(url, path=path)
            model = cls.build_from_pretrained(path)

            return model
        else:
            raise FileNotFoundError(f"{pretrained_model_name_or_path} does not exist.")


class SincNet(nn.Module):
    def __init__(
        self,
        num_features: List[int],
        kernel_size: Union[List[_size_1_t], _size_1_t],
        pool_size: Union[List[_size_1_t], _size_1_t],
        sample_rate: int = 16000,
        stride: _size_1_t = 10,
    ) -> None:
        super().__init__()

        num_layers = len(num_features) - 1

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * num_layers

        if not isinstance(pool_size, list):
            pool_size = [pool_size] * num_layers

        assert len(kernel_size) == num_layers
        assert len(pool_size) == num_layers

        self.norm1d = nn.InstanceNorm1d(num_features[0], affine=True)

        backbone = []

        for layer_index in range(num_layers):
            in_channels = num_features[layer_index]
            out_channels = num_features[layer_index + 1]
            _kernel_size = kernel_size[layer_index]
            _pool_size = pool_size[layer_index]

            if layer_index == 0:
                block = QuadratureSincConvBlock1d(
                    in_channels,
                    out_channels,
                    kernel_size=_kernel_size,
                    pool_size=_pool_size,
                    stride=stride,
                    sample_rate=sample_rate,
                )
            else:
                block = ConvBlock1d(
                    in_channels,
                    out_channels,
                    kernel_size=_kernel_size,
                    pool_size=_pool_size,
                    stride=1,
                )

            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.norm1d(input)

        for block in self.backbone:
            x = block(x)

        output = x

        return output


class SpeakerSegmenterBackbone(nn.Module):
    """Recurrent and projection backbone for frame-level segmentation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_rnn_layers: int = 4,
        num_proj_layers: int = 2,
        dropout: float = 0.5,
        is_causal: bool = False,
    ) -> None:
        """Initialize the recurrent backbone.

        Args:
            in_channels: Number of input features per frame.
            out_channels: Hidden size of each recurrent direction and output
                size of each projection layer.
            num_rnn_layers: Number of recurrent layers.
            num_proj_layers: Number of frame-wise projection layers.
            dropout: Dropout probability between recurrent layers.
            is_causal: If ``True``, use a unidirectional recurrent network;
                otherwise, use a bidirectional network.

        """
        super().__init__()

        if is_causal:
            bidirectional = False
            _out_channels = out_channels
        else:
            bidirectional = True
            _out_channels = 2 * out_channels

        self.rnn = nn.LSTM(
            in_channels,
            out_channels,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        layers = []

        for layer_index in range(num_proj_layers):
            if layer_index == 0:
                layers.append(LinearBlock(_out_channels, out_channels))
            else:
                layers.append(LinearBlock(out_channels, out_channels))

        self.proj = nn.ModuleList(layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of SpeakerSegmenterBackbone.

        Args:
            input: Encoded feature tensor of shape
                (batch_size, in_channels, num_frames).

        Returns:
            torch.Tensor: Output of shape (batch_size, out_channels, num_frames).

        """
        self.rnn.flatten_parameters()

        x = input.permute(0, 2, 1)
        x, _ = self.rnn(x)

        for layer in self.proj:
            x = layer(x)

        output = x.permute(0, 2, 1)

        return output


class LinearBlock(nn.Module):
    """Frame-wise linear projection followed by a LeakyReLU activation."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the projection block.

        Args:
            in_features: Number of input features per frame.
            out_features: Number of output features per frame.

        """
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of LinearBlock.

        Args:
            input (torch.Tensor): Input feature of shape (*, in_features).

        Returns:
            torch.Tensor: Input feature of shape (*, out_features).

        """
        x = self.linear(input)
        output = self.activation(x)

        return output
