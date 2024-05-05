import os

import torch
from omegaconf import DictConfig

import audyn
from audyn.functional.clustering import initialize_centroids, online_kmeans_clustering
from audyn.utils import instantiate, setup_system


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    # TODO: parallel processing
    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    clustering_feature_dir = config.preprocess.clustering_feature_dir
    centroids_path = config.preprocess.centroids_path
    centroids_key = config.preprocess.centroids_key
    max_workers = config.preprocess.max_workers

    assert list_path is not None, "Specify preprocess.list_path."
    assert clustering_feature_dir is not None, "Specify preprocess.clustering_feature_dir."
    assert centroids_path is not None, "Specify preprocess.centroids_path."
    assert centroids_key is not None, "Specify preprocess.centroids_key."
    assert max_workers is not None, "Specify preprocess.max_workers."

    if dump_format != "webdataset":
        raise ValueError("Only webdataset is supported as dump_format.")

    clustering_config = config.preprocess.kmeans.clustering

    # clustering
    dataset = instantiate(clustering_config.dataset)
    dataloader = instantiate(clustering_config.dataloader, dataset)

    n_frames = 0
    batched_mfcc = []
    centroids = None
    num_accumulated_assignments = None

    for named_input in dataloader:
        assert len(named_input["__key__"]) == 1, "Batch size is expected to be 1."

        mfcc = named_input["mfcc"][0]

        n_frames += mfcc.size(-1)
        batched_mfcc.append(mfcc.squeeze(dim=0))

        if n_frames > clustering_config.batch_size:
            batched_mfcc = torch.cat(batched_mfcc, dim=-1)
            batched_mfcc = batched_mfcc.transpose(1, 0)

            if centroids is None:
                centroids = initialize_centroids(
                    batched_mfcc,
                    num_clusters=config.data.clustering.num_clusters,
                    seed=config.system.seed,
                )

            _, centroids, num_accumulated_assignments = online_kmeans_clustering(
                batched_mfcc,
                centroids=centroids,
                num_accumulated_assignments=num_accumulated_assignments,
            )
            batched_mfcc = []

    centroids_dir = os.path.dirname(centroids_path)

    os.makedirs(centroids_dir, exist_ok=True)

    data = {
        centroids_key: centroids,
    }
    torch.save(data, centroids_path)


if __name__ == "__main__":
    main()
