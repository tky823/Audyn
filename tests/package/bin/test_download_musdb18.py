import tempfile

import pytest
from omegaconf import OmegaConf

from audyn.bin.download_musdb18 import download_musdb18


@pytest.mark.slow
def test_download_musdb18_7s() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        config = OmegaConf.create(
            {
                "type": "7s",
                "root": temp_dir,
                "musdb18_root": temp_dir,
                "chunk_size": 16384,
                "unpack": None,
            }
        )
        download_musdb18(config)
