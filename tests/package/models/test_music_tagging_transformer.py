import os

import torch
import torch.nn.functional as F
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir

from audyn.models.ast import HeadTokensAggregator, MLPHead
from audyn.models.music_tagging_transformer import MusicTaggingTransformer
from audyn.modules.music_tagging_transformer import (
    MusicTaggingTransformerEncoder,
    PositionalPatchEmbedding,
)
from audyn.transforms.music_tagging_transformer import (
    MusicTaggingTransformerMelSpectrogram,
)
from audyn.utils._github import download_file_from_github_release


def test_official_music_tagging_transformer() -> None:
    transform = MusicTaggingTransformerMelSpectrogram.build_from_pretrained()
    teacher = MusicTaggingTransformer.build_from_pretrained("music-tagging-transformer_teacher")
    student = MusicTaggingTransformer.build_from_pretrained("music-tagging-transformer_student")

    num_parameters = 0

    for p in teacher.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    assert num_parameters == 4567476

    num_parameters = 0

    for p in student.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    assert num_parameters == 1108788

    transform.eval()
    teacher.eval()
    student.eval()

    url = "https://github.com/tky823/Audyn/releases/download/v0.0.2/test_official_music-tagging-transformer_teacher.pth"  # noqa: E501
    path = os.path.join(
        audyn_test_cache_dir, "test_official_music-tagging-transformer_teacher.pth"
    )
    download_file_from_github_release(url, path)

    data = torch.load(path, weights_only=True)

    waveform = data["input"]
    expected_teacher_output = data["output"]

    with torch.no_grad():
        spectrogram = transform(waveform)
        teacher_logit = teacher(spectrogram)
        teacher_output = F.sigmoid(teacher_logit)

    allclose(teacher_output, expected_teacher_output)

    url = "https://github.com/tky823/Audyn/releases/download/v0.0.2/test_official_music-tagging-transformer_student.pth"  # noqa: E501
    path = os.path.join(
        audyn_test_cache_dir, "test_official_music-tagging-transformer_student.pth"
    )
    download_file_from_github_release(url, path)

    data = torch.load(path, weights_only=True)

    waveform = data["input"]
    expected_student_output = data["output"]

    with torch.no_grad():
        spectrogram = transform(waveform)
        student_logit = student(spectrogram)
        student_output = F.sigmoid(student_logit)

    allclose(student_output, expected_student_output)


def test_music_tagging_transformer() -> None:
    torch.manual_seed(0)

    d_model, dim_feedforward, hidden_channels = 16, 8, 32
    kernel_size = 3
    nhead = 2
    num_layers = 2
    activation = "gelu"
    insert_cls_token = True
    insert_dist_token = False
    batch_first = True

    batch_size = 4
    n_bins, max_frames = 32, 50
    num_classes = 3

    embedding = PositionalPatchEmbedding(
        d_model,
        hidden_channels,
        n_bins,
        kernel_size=kernel_size,
        insert_cls_token=insert_cls_token,
        insert_dist_token=insert_dist_token,
    )
    backbone = MusicTaggingTransformerEncoder(
        d_model,
        nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        activation=activation,
        batch_first=batch_first,
    )
    aggregator = HeadTokensAggregator(
        insert_cls_token=insert_cls_token,
        insert_dist_token=insert_dist_token,
    )
    head = MLPHead(d_model, num_classes)
    model = MusicTaggingTransformer(embedding, backbone, aggregator=aggregator, head=head)

    input = torch.randn((batch_size, n_bins, max_frames))
    length = torch.randint(max_frames // 2, max_frames, (batch_size,))
    output = model(input, length=length)

    assert output.size() == (batch_size, num_classes)
