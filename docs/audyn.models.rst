audyn.models
============

``audyn.models`` provides several models for audio processing.

Submodules
----------

.. toctree::
   :maxdepth: 1

   audyn.models.wavenet
   audyn.models.waveglow
   audyn.models.hifigan
   audyn.models.ast
   audyn.models.ssast
   audyn.models.passt
   audyn.models.clap
   audyn.models.roformer

Classes
-------

Conv-TasNet
^^^^^^^^^^^

.. autoclass:: audyn.models.ConvTasNet
   :members: forward

DPRNN-TasNet
^^^^^^^^^^^^

.. autoclass:: audyn.models.DPRNNTasNet
   :members: forward

WaveNet
^^^^^^^

.. autoclass:: audyn.models.WaveNet
   :members: forward
   
.. autoclass:: audyn.models.MultiSpeakerWaveNet
   :members: forward

WaveGlow
^^^^^^^^

.. autoclass:: audyn.models.WaveGlow
   :members: forward

.. autoclass:: audyn.models.MultiSpeakerWaveGlow
   :members: forward

Audio spectrogram Transformer (AST) and self-supervised AST (SSAST)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> import torch
    >>> from audyn.models import AudioSpectrogramTransformer
    >>> from audyn.models.ast import MLPHead
    >>> torch.manual_seed(0)
    >>> batch_size, n_bins, n_frames = 4, 128, 512
    >>> model = AudioSpectrogramTransformer.build_from_pretrained("ast-base-stride10")
    >>> print(model)
    AudioSpectrogramTransformer(
      (embedding): PositionalPatchEmbedding(
        (conv2d): Conv2d(1, 768, kernel_size=(16, 16), stride=(10, 10))
        (dropout): Dropout(p=0, inplace=False)
      )
      (backbone): TransformerEncoder(
        (layers): ModuleList(
          (0-11): 12 x TransformerEncoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (dropout1): Dropout(p=0.1, inplace=False)
            (dropout2): Dropout(p=0.1, inplace=False)
            (activation): GELU(approximate='none')
          )
        )
        (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      )
      (aggregator): HeadTokensAggregator(cls_token=True, dist_token=True)
      (head): MLPHead(
        (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (linear): Linear(in_features=768, out_features=527, bias=True)
      )
    )
    >>> input = torch.randn((batch_size, n_bins, n_frames))
    >>> output = model(input)
    >>> print(output.size())
    torch.Size([4, 527])
    >>> # remove pretrained head
    >>> model.head = None
    >>> output = model(input)
    >>> print(output.size())
    torch.Size([4, 768])
    >>> # set customized head to model
    >>> embedding_dim = model.embedding.embedding_dim
    >>> num_classes = 50
    >>> head = MLPHead(embedding_dim, num_classes)
    >>> model.head = head
    >>> output = model(input)
    >>> print(output.size())
    torch.Size([4, 50])
    >>> # or set customized head to build_from_pretrained
    >>> model = AudioSpectrogramTransformer.build_from_pretrained("ast-base-stride10", head=head)
    >>> output = model(input)
    >>> print(output.size())
    torch.Size([4, 50])
    >>> # remove aggregator and pretrained head
    >>> model.aggregator = None
    >>> model.head = None
    >>> output = model(input)
    >>> print(output.size())
    torch.Size([4, 602, 768])  # 1 [CLS], 1 [DIST], and 600 patches

.. autoclass:: audyn.models.AudioSpectrogramTransformer
   :members: forward

.. autoclass:: audyn.models.AST

.. autoclass:: audyn.models.MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel
   :members: forward

.. autoclass:: audyn.models.SelfSupervisedAudioSpectrogramTransformer
   :members: forward

.. autoclass:: audyn.models.SSASTMPM

.. autoclass:: audyn.models.SSAST

Patchout faSt Spectrogram Transformer (PaSST)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> import torch
    >>> from audyn.models import PaSST
    >>> from audyn.models.ast import MLPHead
    >>> torch.manual_seed(0)
    >>> batch_size, n_bins, n_frames = 4, 128, 512
    >>> model = PaSST.build_from_pretrained("passt-base-stride10-struct-ap0.476-swa")
    >>> print(model)
    PaSST(
      (embedding): DisentangledPositionalPatchEmbedding(
        (conv2d): Conv2d(1, 768, kernel_size=(16, 16), stride=(10, 10))
        (dropout): Dropout(p=0, inplace=False)
      )
      (dropout): StructuredPatchout(frequency_drops=4, time_drops=40)
      (backbone): TransformerEncoder(
        (layers): ModuleList(
          (0-11): 12 x TransformerEncoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (linear1): Linear(in_features=768, out_features=3072, bias=True)
            (dropout): Dropout(p=0, inplace=False)
            (linear2): Linear(in_features=3072, out_features=768, bias=True)
            (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (dropout1): Dropout(p=0, inplace=False)
            (dropout2): Dropout(p=0, inplace=False)
            (activation): GELU(approximate='none')
          )
        )
        (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      )
      (aggregator): HeadTokensAggregator(cls_token=True, dist_token=True)
      (head): MLPHead(
        (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (linear): Linear(in_features=768, out_features=527, bias=True)
      )
    )
    >>> input = torch.randn((batch_size, n_bins, n_frames))
    >>> output = model(input)
    >>> print(output.size())
    torch.Size([4, 527])
    >>> # remove pretrained head
    >>> model.head = None
    >>> output = model(input)
    >>> print(output.size())
    torch.Size([4, 768])
    >>> # set customized head to model
    >>> embedding_dim = model.embedding.embedding_dim
    >>> num_classes = 50
    >>> head = MLPHead(embedding_dim, num_classes)
    >>> model.head = head
    >>> output = model(input)
    >>> print(output.size())
    torch.Size([4, 50])
    >>> # or set customized head to build_from_pretrained
    >>> model = PaSST.build_from_pretrained("passt-base-stride10-struct-ap0.476-swa", head=head)
    >>> output = model(input)
    >>> print(output.size())
    torch.Size([4, 50])
    >>> # remove aggregator and pretrained head
    >>> model.aggregator = None
    >>> model.head = None
    >>> output = model(input)
    >>> print(output.size())
    torch.Size([4, 82, 768])  # Patchout is applied during training.
    >>> model.eval()
    >>> output = model(input)
    >>> print(output.size())
    torch.Size([4, 602, 768])  # Patchout is not applied during evaluation.

.. autoclass:: audyn.models.PaSST

Contrastive Language-Audio Pretraining (CLAP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> import torch
    >>> from audyn.models import LAIONCLAPAudioEncoder2023
    >>> torch.manual_seed(0)
    >>> batch_size, n_bins, n_frames = 4, 64, 1001
    >>> model = LAIONCLAPAudioEncoder2023.build_from_pretrained("laion-clap-htsat-fused")
    >>> print(model)
    LAIONCLAPAudioEncoder2023(
      (embedding): PatchEmbedding(
        (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2d): Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
        (local_conv2d): Conv2d(1, 96, kernel_size=(4, 12), stride=(4, 12))
        (fusion): FusionBlock(
          (local_attn): LocalAttentionBlock(
            (conv2d1): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1))
            (norm2d1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2d): ReLU()
            (conv2d2): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))
            (norm2d2): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (global_attn): GlobalAttentionBlock(
            (pool2d): AdaptiveAvgPool2d(output_size=1)
            (conv2d1): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1))
            (norm2d1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2d): ReLU()
            (conv2d2): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))
            (norm2d2): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (gate): Sigmoid()
        )
        (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0, inplace=False)
      )
      (backbone): SwinTransformerEncoder(
        (backbone): ModuleList(
          (0): SwinTransformerEncoderBlock(
            (backbone): ModuleList(
              (0-1): 2 x SwinTransformerEncoderLayer(
                (self_attn): SwinRelativePositionalMultiheadAttention(
                  (out_proj): NonDynamicallyQuantizableLinear(in_features=96, out_features=96, bias=True)
                )
                (linear1): Linear(in_features=96, out_features=384, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (linear2): Linear(in_features=384, out_features=96, bias=True)
                (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (dropout1): Dropout(p=0.1, inplace=False)
                (dropout2): Dropout(p=0.1, inplace=False)
                (activation): GELU(approximate='none')
              )
            )
            (downsample): PatchMerge(
              (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (linear): Linear(in_features=384, out_features=192, bias=False)
            )
          )
          (1): SwinTransformerEncoderBlock(
            (backbone): ModuleList(
              (0-1): 2 x SwinTransformerEncoderLayer(
                (self_attn): SwinRelativePositionalMultiheadAttention(
                  (out_proj): NonDynamicallyQuantizableLinear(in_features=192, out_features=192, bias=True)
                )
                (linear1): Linear(in_features=192, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (linear2): Linear(in_features=768, out_features=192, bias=True)
                (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (dropout1): Dropout(p=0.1, inplace=False)
                (dropout2): Dropout(p=0.1, inplace=False)
                (activation): GELU(approximate='none')
              )
            )
            (downsample): PatchMerge(
              (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (linear): Linear(in_features=768, out_features=384, bias=False)
            )
          )
          (2): SwinTransformerEncoderBlock(
            (backbone): ModuleList(
              (0-5): 6 x SwinTransformerEncoderLayer(
                (self_attn): SwinRelativePositionalMultiheadAttention(
                  (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
                )
                (linear1): Linear(in_features=384, out_features=1536, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (linear2): Linear(in_features=1536, out_features=384, bias=True)
                (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (dropout1): Dropout(p=0.1, inplace=False)
                (dropout2): Dropout(p=0.1, inplace=False)
                (activation): GELU(approximate='none')
              )
            )
            (downsample): PatchMerge(
              (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (linear): Linear(in_features=1536, out_features=768, bias=False)
            )
          )
          (3): SwinTransformerEncoderBlock(
            (backbone): ModuleList(
              (0-1): 2 x SwinTransformerEncoderLayer(
                (self_attn): SwinRelativePositionalMultiheadAttention(
                  (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
                )
                (linear1): Linear(in_features=768, out_features=3072, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (linear2): Linear(in_features=3072, out_features=768, bias=True)
                (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout1): Dropout(p=0.1, inplace=False)
                (dropout2): Dropout(p=0.1, inplace=False)
                (activation): GELU(approximate='none')
              )
            )
          )
        )
        (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (aggregator): AverageAggregator()
      (head): MLPHead(
        (linear1): Linear(in_features=768, out_features=512, bias=True)
        (activation): ReLU()
        (linear2): Linear(in_features=512, out_features=512, bias=True)
      )
    )
    >>> spectrogram = torch.randn((batch_size, n_bins, n_frames))
    >>> spectrogram = spectrogram.unsqueeze(dim=-3)
    >>> output = model(spectrogram)
    >>> print(output.size())
    torch.Size([4, 512])

.. autoclass:: audyn.models.LAIONCLAPAudioEncoder2023

.. code-block:: python

    >>> import torch
    >>> from audyn.models import MicrosoftCLAPAudioEncoder2023
    >>> torch.manual_seed(0)
    >>> batch_size, n_bins, n_frames = 4, 64, 965
    >>> model = MicrosoftCLAPAudioEncoder2023.build_from_pretrained("microsoft-clap-2023")
    >>> print(model)
    MicrosoftCLAPAudioEncoder2023(
      (embedding): PatchEmbedding(
        (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2d): Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
        (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0, inplace=False)
      )
      (backbone): SwinTransformerEncoder(
        (backbone): ModuleList(
          (0): SwinTransformerEncoderBlock(
            (backbone): ModuleList(
              (0-1): 2 x SwinTransformerEncoderLayer(
                (self_attn): SwinRelativePositionalMultiheadAttention(
                  (out_proj): NonDynamicallyQuantizableLinear(in_features=96, out_features=96, bias=True)
                )
                (linear1): Linear(in_features=96, out_features=384, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (linear2): Linear(in_features=384, out_features=96, bias=True)
                (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (dropout1): Dropout(p=0.1, inplace=False)
                (dropout2): Dropout(p=0.1, inplace=False)
                (activation): GELU(approximate='none')
              )
            )
            (downsample): PatchMerge(
              (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (linear): Linear(in_features=384, out_features=192, bias=False)
            )
          )
          (1): SwinTransformerEncoderBlock(
            (backbone): ModuleList(
              (0-1): 2 x SwinTransformerEncoderLayer(
                (self_attn): SwinRelativePositionalMultiheadAttention(
                  (out_proj): NonDynamicallyQuantizableLinear(in_features=192, out_features=192, bias=True)
                )
                (linear1): Linear(in_features=192, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (linear2): Linear(in_features=768, out_features=192, bias=True)
                (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (dropout1): Dropout(p=0.1, inplace=False)
                (dropout2): Dropout(p=0.1, inplace=False)
                (activation): GELU(approximate='none')
              )
            )
            (downsample): PatchMerge(
              (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (linear): Linear(in_features=768, out_features=384, bias=False)
            )
          )
          (2): SwinTransformerEncoderBlock(
            (backbone): ModuleList(
              (0-5): 6 x SwinTransformerEncoderLayer(
                (self_attn): SwinRelativePositionalMultiheadAttention(
                  (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
                )
                (linear1): Linear(in_features=384, out_features=1536, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (linear2): Linear(in_features=1536, out_features=384, bias=True)
                (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (dropout1): Dropout(p=0.1, inplace=False)
                (dropout2): Dropout(p=0.1, inplace=False)
                (activation): GELU(approximate='none')
              )
            )
            (downsample): PatchMerge(
              (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (linear): Linear(in_features=1536, out_features=768, bias=False)
            )
          )
          (3): SwinTransformerEncoderBlock(
            (backbone): ModuleList(
              (0-1): 2 x SwinTransformerEncoderLayer(
                (self_attn): SwinRelativePositionalMultiheadAttention(
                  (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
                )
                (linear1): Linear(in_features=768, out_features=3072, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (linear2): Linear(in_features=3072, out_features=768, bias=True)
                (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout1): Dropout(p=0.1, inplace=False)
                (dropout2): Dropout(p=0.1, inplace=False)
                (activation): GELU(approximate='none')
              )
            )
          )
        )
        (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (aggregator): AverageAggregator()
      (head): MicrosoftMLPHead(
        (linear1): Linear(in_features=768, out_features=1024, bias=False)
        (activation): GELU(approximate='none')
        (linear2): Linear(in_features=1024, out_features=1024, bias=False)
        (dropout): Dropout(p=0.5, inplace=False)
        (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
    >>> spectrogram = torch.randn((batch_size, n_bins, n_frames))
    >>> spectrogram = spectrogram.unsqueeze(dim=-3)
    >>> output = model(spectrogram)
    >>> print(output.size())
    torch.Size([4, 1024])

.. autoclass:: audyn.models.MicrosoftCLAPAudioEncoder2023

Rotary Transformer (RoFormer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.models.RoFormerEncoderLayer
.. autoclass:: audyn.models.RoFormerDecoderLayer
.. autoclass:: audyn.models.RoFormerEncoder
.. autoclass:: audyn.models.RoFormerDecoder

Music Tagging Transformer
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: audyn.models.MusicTaggingTransformer

.. code-block:: python

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from audyn.transforms import MusicTaggingTransformerMelSpectrogram
    >>> from audyn.models import MusicTaggingTransformer
    >>> from audyn.models.ast import MLPHead
    >>> from audyn.utils.data.msd_tagging import tags
    >>> torch.manual_seed(0)
    >>> transform = MusicTaggingTransformerMelSpectrogram.build_from_pretrained()
    >>> model = MusicTaggingTransformer.build_from_pretrained("music-tagging-transformer_teacher")
    >>> print(transform)
    MusicTaggingTransformerMelSpectrogram(
      (spectrogram): Spectrogram()
      (mel_scale): MelScale()
      (amplitude_to_db): AmplitudeToDB()
    )
    >>> print(model)
    MusicTaggingTransformer(
      (embedding): PositionalPatchEmbedding(
        (batch_norm): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (backbone): ModuleList(
          (0): ResidualMaxPool2d(
            (backbone): ModuleList(
              (0): ConvBlock2d(
                (conv2d): Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1))
                (batch_norm2d): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu2d): ReLU()
              )
              (1): ConvBlock2d(
                (conv2d): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
                (batch_norm2d): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (post_block2d): ConvBlock2d(
              (conv2d): Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1))
              (batch_norm2d): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (relu2d): ReLU()
            (pool2d): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          )
          (1): ResidualMaxPool2d(
            (backbone): ModuleList(
              (0): ConvBlock2d(
                (conv2d): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
                (batch_norm2d): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu2d): ReLU()
              )
              (1): ConvBlock2d(
                (conv2d): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
                (batch_norm2d): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (relu2d): ReLU()
            (pool2d): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          )
          (2): ResidualMaxPool2d(
            (backbone): ModuleList(
              (0): ConvBlock2d(
                (conv2d): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
                (batch_norm2d): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu2d): ReLU()
              )
              (1): ConvBlock2d(
                (conv2d): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
                (batch_norm2d): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (relu2d): ReLU()
            (pool2d): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
          )
        )
        (linear): Linear(in_features=2048, out_features=256, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (backbone): MusicTaggingTransformerEncoder(
        (layers): ModuleList(
          (0-3): 4 x MusicTaggingTransformerEncoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
            )
            (linear1): Linear(in_features=256, out_features=1024, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
            (linear2): Linear(in_features=1024, out_features=256, bias=True)
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (dropout1): Dropout(p=0.1, inplace=False)
            (dropout2): Dropout(p=0.1, inplace=False)
            (activation): GELU(approximate='none')
          )
        )
      )
      (aggregator): HeadTokensAggregator(cls_token=True)
      (head): MLPHead(
        (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (linear): Linear(in_features=256, out_features=50, bias=True)
      )
    )
    >>> waveform = torch.randn((4, 30 * transform.sample_rate))
    >>> spectrogram = transform(waveform)
    >>> logit = model(spectrogram)
    >>> likelihood = F.sigmoid(logit)
    >>> print(likelihood.size())
    torch.Size([4, 50])
    >>> print(len(tags))
    50  # 50 classes in MSD dataset
    >>> # set customized head to model
    >>> embedding_dim = model.embedding.embedding_dim
    >>> num_classes = 10
    >>> head = MLPHead(embedding_dim, num_classes)
    >>> model = MusicTaggingTransformer.build_from_pretrained("music-tagging-transformer_teacher", head=head)
    >>> logit = model(spectrogram)
    >>> likelihood = F.sigmoid(logit)
    >>> print(likelihood.size())
    torch.Size([4, 10])
