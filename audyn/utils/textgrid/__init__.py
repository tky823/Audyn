from io import TextIOWrapper
from typing import Dict, List

__all__ = ["load_textgrid"]


def load_textgrid(path: str) -> Dict[str, List[Dict[str, str]]]:
    with open(path) as f:
        for line in f:
            line = line.strip("\n")

            if line.startswith("item"):
                break

        alignment = load_data(f)

    return alignment


def load_data(f: TextIOWrapper) -> Dict[str, List[Dict[str, str]]]:
    alignment = {}

    while True:
        line = f.readline()

        if not line:
            break

        _ = f.readline()
        line = f.readline().strip()  # name = "words"
        name = line[8:-1]

        alignment[name] = []

        for _ in range(2):
            _ = f.readline()

        line = f.readline().strip()
        intervals = int(line[18:])

        for _ in range(intervals):
            _ = f.readline()
            line = f.readline().strip()
            start = float(line[7:])
            line = f.readline().strip()
            end = float(line[7:])
            line = f.readline().strip()
            text = line[8:-1]

            alignment[name].append({"start": start, "end": end, "text": text})

    return alignment
