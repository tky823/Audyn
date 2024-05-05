import io
import os
import tarfile
from typing import Dict

import torch
from omegaconf import DictConfig

import audyn
from audyn.utils import instantiate, setup_system


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    feature_dir = config.preprocess.feature_dir
    jsonl_path = config.preprocess.jsonl_path
    max_workers = config.preprocess.max_workers

    assert list_path is not None, "Specify preprocess.list_path."
    assert feature_dir is not None, "Specify preprocess.feature_dir."
    assert jsonl_path is not None, "Specify preprocess.jsonl_path."
    assert max_workers is not None, "Specify preprocess.max_workers."

    if dump_format != "webdataset":
        raise ValueError("Only webdataset is supported as dump_format.")

    feature_extraction_config = config.preprocess.kmeans.feature_extraction
    feature_dir = config.preprocess.feature_dir
    clustering_feature_dir = config.preprocess.clustering_feature_dir

    dataset = instantiate(feature_extraction_config.dataset)
    shards: Dict[str, tarfile.TarFile] = {}

    for named_input in dataset:
        ytid = named_input["__key__"]
        url = named_input["__url__"]
        path_from_feature_dir = os.path.relpath(url, feature_dir)
        clustering_path = os.path.join(clustering_feature_dir, path_from_feature_dir)
        clustering_dir = os.path.dirname(clustering_path)

        if clustering_path not in shards:
            os.makedirs(clustering_dir, exist_ok=True)

            shards[clustering_path] = tarfile.open(clustering_path, mode="w")

        binary = io.BytesIO()
        torch.save(named_input["mfcc"], binary)
        binary.seek(0)

        tarinfo = tarfile.TarInfo(f"{ytid}.mfcc.pth")
        tarinfo.size = len(binary.getvalue())
        shards[clustering_path].addfile(tarinfo, binary)

    for shard in shards.values():
        shard.close()


if __name__ == "__main__":
    main()
