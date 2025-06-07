import os

from omegaconf import DictConfig
from tqdm import tqdm

import audyn
from audyn.utils import setup_config


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    feature_dir = config.preprocess.feature_dir
    fma_root = config.preprocess.fma_root
    subset = config.preprocess.subset

    assert dump_format is not None, "Specify preprocess.dump_format."
    assert list_path is not None, "Specify preprocess.list_path."
    assert feature_dir is not None, "Specify preprocess.feature_dir."
    assert subset is not None, "Specify preprocess.subset."

    if dump_format == "fma-small_nafp":
        with open(list_path) as f:
            for line in tqdm(f):
                filename = line.strip()
                process(
                    filename=filename,
                    fma_root=fma_root,
                    feature_dir=feature_dir,
                    subset=subset,
                )
    else:
        raise NotImplementedError("Only dump_format=fma-small_nafp is supported.")


def process(filename: str, fma_root: str, feature_dir: str, subset: str, ext: str = "wav") -> None:
    sub_index = os.path.dirname(filename)
    filename = os.path.basename(filename)

    if subset == "test":
        path = os.path.join(
            fma_root,
            "music",
            "test-query-db-500-30s",
            "db",
            sub_index,
            f"{filename}.{ext}",
        )
    else:
        raise ValueError(f"Unsupported subset {subset} is found.")

    _feature_dir = os.path.join(feature_dir, "db", sub_index)

    os.makedirs(_feature_dir, exist_ok=True)

    _current_dir = os.path.abspath(os.curdir)
    os.chdir(_feature_dir)
    _rel_path = os.path.relpath(path, _feature_dir)

    assert os.path.exists(_rel_path)

    os.symlink(_rel_path, f"{filename}.{ext}")
    os.chdir(_current_dir)


if __name__ == "__main__":
    main()
