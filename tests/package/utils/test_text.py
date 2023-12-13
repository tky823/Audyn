import os
import tempfile

import pytest

from audyn.utils.data.cmudict import CMUDict
from audyn.utils.text import load_text
from audyn.utils.text.pronunciation import Phonemizer


def test_load_text() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        text_path = os.path.join(temp_dir, "dummy.txt")

        # end with \n
        with open(text_path, mode="w") as f:
            f.write("All in the golden afternoon\n")

        text = load_text(text_path)

        assert text == "All in the golden afternoon"

        # end without \n
        with open(text_path, mode="w") as f:
            f.write("All in the golden afternoon")

        text = load_text(text_path)

        assert text == "All in the golden afternoon"

        # error case: multiple lines
        with open(text_path, mode="w") as f:
            f.write("All in the golden afternoon\n")
            f.write("Full leisurely we glide;\n")

        with pytest.raises(ValueError) as e:
            text = load_text(text_path)

            assert str(e.value) == f"The text format of {text_path} is incorrect."


@pytest.mark.parametrize("dict_type", ["dummy", "cmu_dict"])
def test_phonemizer(dict_type: str) -> None:
    if dict_type == "dummy":
        pron_dict = {
            "apple": ["AE1", "P", "AH0", "L"],
            "ate": ["EY1", "T"],
            "i": ["AY1"],
        }
    else:
        pron_dict = CMUDict()

    phonemizer = Phonemizer(pron_dict)
    tokens = ["i", "ate", "an", "apple"]
    phonemes = phonemizer(tokens)

    if dict_type == "dummy":
        assert phonemes == ["AY1", "EY1", "T", "<UNK>", "AE1", "P", "AH0", "L"]
    else:
        assert phonemes == ["AY1", "EY1", "T", "AE1", "N", "AE1", "P", "AH0", "L"]
