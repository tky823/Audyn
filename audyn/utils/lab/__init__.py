from typing import List

__all__ = ["load_mfa_lab"]


def load_mfa_lab(path: str) -> List[str]:
    with open(path) as f:
        lines = f.readlines()

    assert len(lines) > 0, f"{path} is empty."
    assert len(lines) == 1, f"{path} is too long. Deteceted {len(lines)} lines."

    line = lines[0]
    line = line.strip("\n")
    text = line.split(" ")

    return text
