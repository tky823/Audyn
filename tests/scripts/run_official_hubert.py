import torch
from datasets import load_dataset
from transformers import AutoProcessor, HubertModel


def main() -> None:
    torch.manual_seed(0)

    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_demo",
        "clean",
        split="validation",
        trust_remote_code=True,
    )
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    inputs = processor(
        dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt"
    )

    input = inputs["input_values"]
    input = input.squeeze(dim=0)
    input = input.unsqueeze(dim=0)

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    output = outputs.last_hidden_state
    output = output.squeeze()

    data = {
        "input": input,
        "output": output,
    }

    torch.save(data, "test_official_hubert.pth")


if __name__ == "__main__":
    main()
