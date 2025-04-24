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
    discrete_feature_dir = config.preprocess.discrete_feature_dir
    centroids_path = config.preprocess.centroids_path
    clustering_feature_key = config.data.clustering.feature
    discrete_feature_key = config.data.clustering.discrete_feature
    centroids_key = config.preprocess.centroids_key
    max_workers = config.preprocess.max_workers

    assert list_path is not None, "Specify preprocess.list_path."
    assert feature_dir is not None, "Specify preprocess.feature_dir."
    assert discrete_feature_dir is not None, "Specify preprocess.discrete_feature_dir."
    assert max_workers is not None, "Specify preprocess.max_workers."

    if dump_format != "webdataset":
        raise ValueError("Only webdataset is supported as dump_format.")

    feature_extraction_config = config.preprocess.kmeans.feature_extraction

    dataset = instantiate(feature_extraction_config.dataset)
    shards: Dict[str, tarfile.TarFile] = {}

    # centroids: (num_clusters, num_features)
    data = torch.load(
        centroids_path,
        map_location=lambda storage, loc: storage,
        weights_only=True,
    )
    centroids = data[centroids_key]

    for named_input in dataset:
        ytid = named_input["__key__"]
        url = named_input["__url__"]
        clustering_feature = named_input[clustering_feature_key]
        norm = torch.sum(centroids**2, dim=-1, keepdim=True)
        dot = torch.matmul(centroids, clustering_feature)
        discrete_idx = torch.argmin(norm - 2 * dot, dim=0)
        path_from_feature_dir = os.path.relpath(url, feature_dir)
        discrete_path = os.path.join(discrete_feature_dir, path_from_feature_dir)
        discrete_dir = os.path.dirname(discrete_path)

        if discrete_path not in shards:
            os.makedirs(discrete_dir, exist_ok=True)

            shards[discrete_path] = tarfile.open(discrete_path, mode="w")

        binary = io.BytesIO()
        torch.save(discrete_idx, binary)
        binary.seek(0)

        tarinfo = tarfile.TarInfo(f"{ytid}.{discrete_feature_key}.pth")
        tarinfo.size = len(binary.getvalue())
        shards[discrete_path].addfile(tarinfo, binary)

    for shard in shards.values():
        shard.close()


if __name__ == "__main__":
    main()
