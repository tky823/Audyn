import os
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from packaging import version
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single

from ..modules.encodec import DecoderBlock, EncoderBlock, _pad1d
from ..modules.transformer import get_activation
from .rvqvae import RVQVAE

__all__ = [
    "EnCodec",
    "Encoder",
    "Decoder",
]

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class EnCodec(RVQVAE):
    r"""EnCodec using residual vector quantizer.

    Args:
        encoder (nn.Module): Encoder which returns latent feature of
            shape (batch_size, embedding_dim, \*).
        decoder (nn.Module): Decoder which takes latent feature of
            shape (batch_size, embedding_dim, \*).
        codebook_size (int): Size of codebook.
        embedding_dim (int): Number of embedding dimension.
        num_stages (int): Number of residual stages of RVQ.
        dropout (bool): Dropout of RVQ. Default: ``True``.
        init_by_kmeans (int): Number of iterations in k-means clustering initialization.
            If non-positive value is given, k-means clustering initialization is not used.
        seed (int): Random seed for k-means clustering initialization.

    """

    def forward(
        self,
        input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Forward pass of EnCodec.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, *input_shape).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Reconstructed feature of same shape as input.
                - torch.Tensor: Latent feature of shape \
                    (batch_size, embedding_dim, *latent_shape). In most cases, latent_shape is \
                    smaller than input_shape.
                - torch.Tensor: Quantized latent feature of shape \
                    (batch_size, num_stages, embedding_dim, *latent_shape).
                - torch.Tensor: Indices of embeddings in codebook of shape \
                    (batch_size, num_stages, *latent_shape).

        .. note::

            Gradient from decoder does not back propagate to codebook.

        """
        orig_length = input.size(-1)

        encoded = self.encode(input)
        hierarchical_quantized, hierarchical_residual, indices = self.quantize(encoded)
        quantized_straight_through = hierarchical_residual + torch.detach(
            hierarchical_quantized - hierarchical_residual
        )
        quantized_straight_through = quantized_straight_through.sum(dim=1)
        x = self.decode(quantized_straight_through, stage_wise=False)
        padding = x.size(-1) - orig_length
        output = F.pad(x, (0, -padding))

        return output, encoded, hierarchical_quantized, hierarchical_residual, indices

    @property
    def down_scale(self) -> Optional[int]:
        """Try to find down_scale or downscale parameter from self.encoder."""
        encoder = self.encoder
        scale = 1

        if isinstance(encoder, Encoder):
            encoder: Encoder

            for s in encoder.stride:
                scale *= s
        elif hasattr(encoder, "down_scale") and not callable(encoder.down_scale):
            scale = encoder.down_scale
        elif hasattr(encoder, "downscale") and not callable(encoder.downscale):
            scale = encoder.downscale
        else:
            scale = None

        return scale

    @property
    def up_scale(self) -> Optional[int]:
        """Try to find up_scale or upscale parameter from self.decoder."""
        decoder = self.decoder
        scale = 1

        if isinstance(decoder, Decoder):
            decoder: Decoder

            for s in decoder.stride:
                scale *= s
        elif hasattr(decoder, "up_scale") and not callable(decoder.up_scale):
            scale = decoder.up_scale
        elif hasattr(decoder, "upscale") and not callable(decoder.upscale):
            scale = decoder.upscale
        else:
            scale = None

        return scale

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
    ) -> "EnCodec":
        """Build EnCodec from pretrained model.

        Args:
            pretrained_model_name_or_path (str): Path to pretrained model or name of
                pretrained model.

        Returns:
            EnCodec: Built EnCodec model.

        Examples:

            >>> from audyn.models import EnCodec
            >>> model = EnCodec.build_from_pretrained("encodec_24khz")

        .. note::

            Supported pretrained model names are
                - encodec_24khz

        """  # noqa: E501
        from ..utils._github import download_file_from_github_release
        from ..utils._hydra.utils import instantiate

        pretrained_model_configs = _create_pretrained_encodec_configs()

        if os.path.exists(pretrained_model_name_or_path):
            state_dict = torch.load(
                pretrained_model_name_or_path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
            model_state_dict: OrderedDict = state_dict["model"]
            resolved_config = state_dict["resolved_config"]
            resolved_config = OmegaConf.create(resolved_config)
            pretrained_model_config = resolved_config.model
            pretrained_model_config["_target_"] = f"{cls.__module__}.{cls.__name__}"
            model: EnCodec = instantiate(pretrained_model_config)
            # For backward compatibility, weight_norm is removed here.
            model.encoder.remove_weight_norm_()
            model.decoder.remove_weight_norm_()
            model.load_state_dict(model_state_dict)
            model.encoder.weight_norm_()
            model.decoder.weight_norm_()

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


class Encoder(nn.Module):
    """Encoder of EnCodec."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        depth_rate: int,
        kernel_size_in: _size_1_t,
        kernel_size_out: _size_1_t,
        kernel_size: _size_1_t,
        stride: List[_size_1_t],
        dilation_rate: int = 1,
        padding_mode: str = "reflect",
        num_unit_layers: int = 1,
        rnn_type: Union[str, Type] = "lstm",
        num_rnn_layers: int = 2,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.elu,
        weight_regularization: Optional[str] = "weight_norm",
        is_causal: bool = True,
    ) -> None:
        super().__init__()

        self.conv1d_in = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size_in,
            stride=1,
        )

        _in_channels = hidden_channels
        backbone = []

        for _stride in stride:
            _out_channels = depth_rate * _in_channels

            block = EncoderBlock(
                _in_channels,
                _out_channels,
                kernel_size=kernel_size,
                stride=_stride,
                dilation_rate=dilation_rate,
                padding_mode=padding_mode,
                num_layers=num_unit_layers,
                activation=activation,
                weight_regularization=weight_regularization,
                is_causal=is_causal,
            )
            backbone.append(block)
            _in_channels = _out_channels

        self.backbone = nn.Sequential(*backbone)

        rnn_cls = _get_rnn(rnn_type)

        if is_causal:
            self.rnn = rnn_cls(
                _out_channels,
                _out_channels,
                num_layers=num_rnn_layers,
                batch_first=True,
                bidirectional=False,
            )
        else:
            self.rnn = rnn_cls(
                _out_channels,
                _out_channels // 2,
                num_layers=num_rnn_layers,
                batch_first=True,
                bidirectional=True,
            )

        if isinstance(activation, str):
            activation = get_activation(activation)

        self.nonlinear1d = activation
        self.conv1d_out = nn.Conv1d(
            _out_channels,
            out_channels,
            kernel_size=kernel_size_out,
            stride=1,
        )

        self.kernel_size_in = _single(kernel_size_in)
        self.kernel_size_out = _single(kernel_size_out)
        self.stride = stride
        self.padding_mode = padding_mode
        self.is_causal = is_causal

        self.registered_weight_norms = set()

        if weight_regularization is None:
            pass
        elif weight_regularization == "weight_norm":
            self.registered_weight_norms.add("backbone")
            self.weight_norm_()
        else:
            raise ValueError(
                "{}-based weight regularization is not supported.".format(weight_regularization)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of Encoder.

        Args:
            input (torch.Tensor): Waveform of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Compressed feature of shape (batch_size, out_channels, length').

        """
        (kernel_size_in,) = self.kernel_size_in
        (kernel_size_out,) = self.kernel_size_out

        x = _pad1d(
            input,
            kernel_size=kernel_size_in,
            stride=1,
            mode=self.padding_mode,
            is_causal=self.is_causal,
        )
        x = self.conv1d_in(x)
        x = self.backbone(x)
        x = x.permute(0, 2, 1).contiguous()
        residual = x
        x, _ = self.rnn(residual)
        x = x + residual
        x = x.permute(0, 2, 1).contiguous()
        x = self.nonlinear1d(x)
        x = _pad1d(
            x,
            kernel_size=kernel_size_out,
            stride=1,
            mode=self.padding_mode,
            is_causal=self.is_causal,
        )
        output = self.conv1d_out(x)

        return output

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        if "backbone" not in self.registered_weight_norms:
            for block in self.backbone:
                block: EncoderBlock
                block.weight_norm_()

            self.registered_weight_norms.add("backbone")

        self.conv1d_in = weight_norm_fn(self.conv1d_in)
        self.conv1d_out = weight_norm_fn(self.conv1d_out)
        self.registered_weight_norms.add("conv1d_in")
        self.registered_weight_norms.add("conv1d_out")

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        for block in self.backbone:
            block: EncoderBlock
            block.remove_weight_norm_()

        self.registered_weight_norms.remove("backbone")

        self.conv1d_in = remove_weight_norm_fn(self.conv1d_in, *remove_weight_norm_args)
        self.conv1d_out = remove_weight_norm_fn(self.conv1d_out, *remove_weight_norm_args)
        self.registered_weight_norms.remove("conv1d_in")
        self.registered_weight_norms.remove("conv1d_out")


class Decoder(nn.Module):
    """Decoder of EnCodec."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        depth_rate: int,
        kernel_size_in: _size_1_t,
        kernel_size_out: _size_1_t,
        kernel_size: _size_1_t,
        stride: List[_size_1_t],
        dilation_rate: int = 1,
        padding_mode: str = "reflect",
        num_unit_layers: int = 1,
        rnn_type: Union[str, Type] = "lstm",
        num_rnn_layers: int = 2,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.elu,
        weight_regularization: Optional[str] = "weight_norm",
        is_causal: bool = True,
    ) -> None:
        super().__init__()

        _out_channels = hidden_channels * (depth_rate ** len(stride))

        self.conv1d_in = nn.Conv1d(
            in_channels,
            _out_channels,
            kernel_size=kernel_size_in,
            stride=1,
        )

        rnn_cls = _get_rnn(rnn_type)

        if is_causal:
            self.rnn = rnn_cls(
                _out_channels,
                _out_channels,
                num_layers=num_rnn_layers,
                batch_first=True,
                bidirectional=False,
            )
        else:
            self.rnn = rnn_cls(
                _out_channels,
                _out_channels // 2,
                num_layers=num_rnn_layers,
                batch_first=True,
                bidirectional=True,
            )

        _in_channels = _out_channels
        backbone = []

        for _stride in stride:
            _out_channels = _in_channels // depth_rate

            block = DecoderBlock(
                _in_channels,
                _out_channels,
                kernel_size=kernel_size,
                stride=_stride,
                dilation_rate=dilation_rate,
                padding_mode=padding_mode,
                num_layers=num_unit_layers,
                activation=activation,
                weight_regularization=weight_regularization,
                is_causal=is_causal,
            )
            backbone.append(block)
            _in_channels = _out_channels

        self.backbone = nn.Sequential(*backbone)

        if isinstance(activation, str):
            activation = get_activation(activation)

        self.nonlinear1d = activation
        self.conv1d_out = nn.Conv1d(
            _out_channels,
            out_channels,
            kernel_size=kernel_size_out,
            stride=1,
        )

        self.kernel_size_in = _single(kernel_size_in)
        self.kernel_size_out = _single(kernel_size_out)
        self.padding_mode = padding_mode
        self.is_causal = is_causal

        self.registered_weight_norms = set()

        if weight_regularization is None:
            pass
        elif weight_regularization == "weight_norm":
            self.registered_weight_norms.add("backbone")
            self.weight_norm_()
        else:
            raise ValueError(
                "{}-based weight regularization is not supported.".format(weight_regularization)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of Decoder.

        Args:
            input (torch.Tensor): Compressed feature of shape (batch_size, in_channels, length).
            denoise (bool): Denoising flag.

        Returns:
            torch.Tensor: Reconstructed waveform of shape (batch_size, out_channels, length').

        """
        (kernel_size_in,) = self.kernel_size_in
        (kernel_size_out,) = self.kernel_size_out

        x = _pad1d(
            input,
            kernel_size=kernel_size_in,
            stride=1,
            mode=self.padding_mode,
            is_causal=self.is_causal,
        )
        x = self.conv1d_in(x)
        x = x.permute(0, 2, 1).contiguous()
        residual = x
        x, _ = self.rnn(x)
        x = x + residual
        x = x.permute(0, 2, 1).contiguous()
        x = self.backbone(x)
        x = self.nonlinear1d(x)
        x = _pad1d(
            x,
            kernel_size=kernel_size_out,
            stride=1,
            mode=self.padding_mode,
            is_causal=self.is_causal,
        )
        output = self.conv1d_out(x)

        return output

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        if "backbone" not in self.registered_weight_norms:
            for block in self.backbone:
                block: DecoderBlock
                block.weight_norm_()

            self.registered_weight_norms.add("backbone")

        self.conv1d_in = weight_norm_fn(self.conv1d_in)
        self.conv1d_out = weight_norm_fn(self.conv1d_out)
        self.registered_weight_norms.add("conv1d_in")
        self.registered_weight_norms.add("conv1d_out")

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        for block in self.backbone:
            block: EncoderBlock
            block.remove_weight_norm_()

        self.registered_weight_norms.remove("backbone")

        self.conv1d_in = remove_weight_norm_fn(self.conv1d_in, *remove_weight_norm_args)
        self.conv1d_out = remove_weight_norm_fn(self.conv1d_out, *remove_weight_norm_args)
        self.registered_weight_norms.remove("conv1d_in")
        self.registered_weight_norms.remove("conv1d_out")


def _get_rnn(rnn_type: Union[str, Type]) -> Type:
    if isinstance(rnn_type, type):
        return rnn_type
    elif isinstance(rnn_type, str):
        if rnn_type.lower() == "lstm":
            return nn.LSTM
        else:
            raise ValueError(f"{rnn_type} is not supported.")
    else:
        raise ValueError(f"{rnn_type} is not supported.")


def _create_pretrained_encodec_configs() -> Dict[str, Dict[str, str]]:
    """Create config of pretrained EnCodec models."""
    from ..utils import model_cache_dir

    configs = {
        "encodec_24khz": {
            "url": "https://github.com/tky823/Audyn/releases/download/v0.2.1/encodec_24kHz.pth",  # noqa: E501
            "path": os.path.join(
                model_cache_dir,
                "EnCodec",
                "7f737d31",
                "encodec_24khz.pth",
            ),
            "sha256": "7f737d318e0dac4d140cad0f38a7184128c795cfebf764341e631d381ae33c3c",
        },
    }

    return configs
