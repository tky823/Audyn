import pytest
import torch
import torch.nn as nn

from audyn.models.ast import HeadTokensAggregator, MLPHead
from audyn.models.passt import DisentangledPositionalPatchEmbedding, PaSST
from audyn.modules.passt import UnstructuredPatchout


@pytest.mark.parametrize("sample_wise", [True, False])
def test_passt(sample_wise: bool) -> None:
    torch.manual_seed(0)

    d_model, out_channels = 8, 10
    n_bins, n_frames = 8, 30
    kernel_size = (4, 4)
    insert_cls_token, insert_dist_token = True, True
    batch_size = 4

    nhead = 2
    dim_feedforward = 5
    num_layers = 2
    num_drops = 8

    patch_embedding = DisentangledPositionalPatchEmbedding(
        d_model,
        kernel_size=kernel_size,
        insert_cls_token=insert_cls_token,
        insert_dist_token=insert_dist_token,
        n_bins=n_bins,
        n_frames=n_frames,
    )
    norm = nn.LayerNorm(d_model)
    dropout = UnstructuredPatchout(
        num_drops=num_drops,
        sample_wise=sample_wise,
    )
    encoder_layer = nn.TransformerEncoderLayer(
        d_model,
        nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
    )
    transformer = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_layers,
        norm=norm,
    )
    aggregator = HeadTokensAggregator(
        insert_cls_token=insert_cls_token,
        insert_dist_token=insert_dist_token,
    )
    head = MLPHead(d_model, out_channels)

    model = PaSST(
        patch_embedding,
        dropout,
        transformer,
        aggregator=aggregator,
        head=head,
    )

    input = torch.randn((batch_size, n_bins, n_frames))
    output = model(input)

    assert output.size() == (batch_size, out_channels)
