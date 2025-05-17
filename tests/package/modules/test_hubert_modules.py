import torch

from audyn.modules.hubert import HuBERTGELU


def test_hubert_gelu() -> None:
    torch.manual_seed(0)

    model = HuBERTGELU()

    input = torch.randn((2, 3, 4))

    output = model(input)

    assert output.size() == input.size()
