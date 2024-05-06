import glob
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

from omegaconf import DictConfig
from tqdm import tqdm

import audyn
from audyn.utils import setup_system


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    feature_dir = config.preprocess.feature_dir
    discrete_feature_dir = config.preprocess.discrete_feature_dir
    unified_feature_dir = config.preprocess.unified_feature_dir
    discrete_feature_key = config.data.clustering.discrete_feature
    max_workers = config.preprocess.max_workers

    assert list_path is not None, "Specify preprocess.list_path."
    assert feature_dir is not None, "Specify preprocess.feature_dir."
    assert discrete_feature_dir is not None, "Specify preprocess.discrete_feature_dir."
    assert unified_feature_dir is not None, "Specify preprocess.unified_feature_dir."
    assert max_workers is not None, "Specify preprocess.max_workers."

    if dump_format != "webdataset":
        raise ValueError("Only webdataset is supported as dump_format.")

    template_path = os.path.join(feature_dir, "*.tar")
    urls = sorted(glob.glob(template_path))

    assert len(glob.glob(os.path.join(discrete_feature_dir, "*.tar"))) == len(urls)

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for url in urls:
                path = os.path.relpath(url, feature_dir)
                future = executor.submit(
                    process,
                    path=path,
                    feature_dir=feature_dir,
                    discrete_feature_dir=discrete_feature_dir,
                    unified_feature_dir=unified_feature_dir,
                    discrete_feature_key=discrete_feature_key,
                )
                futures.append(future)

            for future in tqdm(futures):
                future.result()
    else:
        for url in urls:
            path = os.path.relpath(url, feature_dir)
            process(
                path=path,
                feature_dir=feature_dir,
                discrete_feature_dir=discrete_feature_dir,
                unified_feature_dir=unified_feature_dir,
                discrete_feature_key=discrete_feature_key,
            )


def process(
    path: str,
    feature_dir: str,
    discrete_feature_dir: str,
    unified_feature_dir: str,
    discrete_feature_key: str = None,
) -> None:
    url = os.path.join(feature_dir, path)
    discrete_path = os.path.join(discrete_feature_dir, path)
    unified_path = os.path.join(unified_feature_dir, path)
    unified_dir = os.path.dirname(unified_path)

    os.makedirs(unified_dir, exist_ok=True)
    shard = tarfile.open(unified_path, mode="w")

    with tarfile.open(url) as f, tarfile.open(discrete_path) as f_discrete:
        ytids = []
        feature_mapping: Dict[str, List[str]] = {}
        discrete_mapping: Dict[str, str] = {}

        for name in f.getnames():
            ytid, *_ = name.split(".")

            if ytid not in ytids:
                ytids.append(ytid)
                feature_mapping[ytid] = []
                discrete_mapping[ytid] = f"{ytid}.{discrete_feature_key}.pth"

            feature_mapping[ytid].append(name)

        for ytid in ytids:
            feature_keys = feature_mapping[ytid]

            for name in feature_keys:
                tarinfo = f.getmember(name)
                binary = f.extractfile(tarinfo)
                shard.addfile(tarinfo, binary)

            tarinfo = f_discrete.getmember(discrete_mapping[ytid])
            binary = f.extractfile(tarinfo)
            shard.addfile(tarinfo, binary)

    shard.close()


if __name__ == "__main__":
    main()
