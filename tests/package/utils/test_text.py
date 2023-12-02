import os
import tempfile

from audyn.utils.text import load_text


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
