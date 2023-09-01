import torch

from audyn.modules.duration_predictor import FastSpeechDurationPredictor


def test_duration_predictor():
    batch_size, max_length = 2, 8
    num_features = [4, 2]
    kernel_size = 3

    length = torch.randint(1, max_length, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = torch.randn((batch_size, max_length, num_features[0]))
    input = input.masked_fill(padding_mask.unsqueeze(dim=-1), 0)

    duration_predictor = FastSpeechDurationPredictor(
        num_features, kernel_size=kernel_size, batch_first=True
    )
    output = duration_predictor(input, padding_mask=padding_mask)

    assert output.size() == (batch_size, max_length)
