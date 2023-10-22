from typing import List

__all__ = ["load_text"]


def load_text(path: str) -> List[str]:
    """Load text file.

    Args:
        path (str): Path to text file.

    Returns:
        list: List of words written in text file.

    .. code-block:: shell

        echo "All in the golden afternoon" > "sample.txt"
        python
        >>> from audyn.utils.text import load_text
        >>> load_text("sample.txt")
        ['All', 'in', 'the', 'golden', 'afternoon']

    """
    with open(path) as f:
        lines = f.readlines()

    if len(lines) != 1:
        raise ValueError(f"The text format of {path} is incorrect.")

    line = lines[0]
    line = line.strip()
    text = line.split(" ")

    return text
