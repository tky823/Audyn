from .ast import AST, AudioSpectrogramTransformer
from .bsrnn import BSRNN, BandSplitRNN
from .clap import (
    LAIONAudioEncoder2023,
    LAIONCLAPAudioEncoder2023,
    MicrosoftAudioEncoder2023,
    MicrosoftCLAPAudioEncoder2023,
)
from .conv_tasnet import ConvTasNet
from .dprnn_tasnet import DPRNNTasNet
from .encodec import EnCodec
from .fastspeech import FastSpeech, MultiSpeakerFastSpeech
from .hifigan import HiFiGANDiscriminator, HiFiGANGenerator, HiFiGANVocoder
from .music_tagging_transformer import (
    MusicTaggingTransformer,
    MusicTaggingTransformerLinearProbing,
)
from .passt import PaSST
from .roformer import (
    RoFormerDecoder,
    RoFormerDecoderLayer,
    RoFormerEncoder,
    RoFormerEncoderLayer,
)
from .rvqvae import RVQVAE
from .soundstream import SoundStream
from .ssast import (
    SSAST,
    SSASTMPM,
    MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel,
    SelfSupervisedAudioSpectrogramTransformer,
)
from .vae import BaseVAE
from .vqvae import VQVAE, GumbelVQVAE
from .waveglow import MultiSpeakerWaveGlow, WaveGlow
from .wavenet import MultiSpeakerWaveNet, WaveNet
from .wavenext import WaveNeXtVocoder

__all__ = [
    # Conv-TasNet
    "ConvTasNet",
    # DPRNN-TasNet
    "DPRNNTasNet",
    # BandSplitRNN
    "BandSplitRNN",
    "BSRNN",
    # WaveNet
    "WaveNet",
    "MultiSpeakerWaveNet",
    # WaveGlow
    "WaveGlow",
    "MultiSpeakerWaveGlow",
    # FastSpeech
    "FastSpeech",
    "MultiSpeakerFastSpeech",
    # HiFi-GAN
    "HiFiGANGenerator",
    "HiFiGANVocoder",
    "HiFiGANDiscriminator",
    # WaveNeXt Vocoder
    "WaveNeXtVocoder",
    # VAE
    "BaseVAE",
    "VQVAE",
    "GumbelVQVAE",
    "RVQVAE",
    "SoundStream",
    "EnCodec",
    # AST
    "AudioSpectrogramTransformer",
    "AST",
    # SSAST
    "MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel",
    "SelfSupervisedAudioSpectrogramTransformer",
    "SSASTMPM",
    "SSAST",
    # PaSST
    "PaSST",
    # CLAP
    "LAIONAudioEncoder2023",
    "LAIONCLAPAudioEncoder2023",
    "MicrosoftCLAPAudioEncoder2023",
    "MicrosoftAudioEncoder2023",
    # RoFormer
    "RoFormerEncoderLayer",
    "RoFormerDecoderLayer",
    "RoFormerEncoder",
    "RoFormerDecoder",
    # Music Tagging Transformer
    "MusicTaggingTransformer",
    "MusicTaggingTransformerLinearProbing",
]
