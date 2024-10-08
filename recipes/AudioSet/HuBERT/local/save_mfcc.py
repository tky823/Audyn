import io
import os
import tarfile
from typing import Dict

import torch
from omegaconf import DictConfig

import audyn
from audyn.utils import instantiate, setup_config


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    # TODO: parallel processing
    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    feature_dir = config.preprocess.feature_dir
    clustering_feature_dir = config.preprocess.clustering_feature_dir
    max_workers = config.preprocess.max_workers

    assert list_path is not None, "Specify preprocess.list_path."
    assert feature_dir is not None, "Specify preprocess.feature_dir."
    assert clustering_feature_dir is not None, "Specify preprocess.clustering_feature_dir."
    assert max_workers is not None, "Specify preprocess.max_workers."

    if dump_format != "webdataset":
        raise ValueError("Only webdataset is supported as dump_format.")

    feature_extraction_config = config.preprocess.kmeans.feature_extraction
    clustering_feature_key = config.data.clustering.feature

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
        torch.save(named_input[clustering_feature_key], binary)
        binary.seek(0)

        tarinfo = tarfile.TarInfo(f"{ytid}.{clustering_feature_key}.pth")
        tarinfo.size = len(binary.getvalue())
        shards[clustering_path].addfile(tarinfo, binary)

    for shard in shards.values():
        shard.close()


if __name__ == "__main__":
    main()
