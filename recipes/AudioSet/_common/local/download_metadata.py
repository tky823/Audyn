import os

from omegaconf import DictConfig

import audyn
from audyn.utils import setup_config
from audyn.utils.data.download import download_file


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    ontology_root = config.preprocess.ontology_root
    label_root = config.preprocess.label_root
    csv_root = config.preprocess.csv_root

    ontology_url = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
    label_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"  # noqa: E501
    balanced_train_csv_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"  # noqa: E501
    unbalanced_train_csv_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"  # noqa: E501
    eval_csv_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"  # noqa: E501

    os.makedirs(ontology_root, exist_ok=True)
    os.makedirs(label_root, exist_ok=True)
    os.makedirs(csv_root, exist_ok=True)

    # ontology
    filename = os.path.basename(ontology_url)
    path = os.path.join(ontology_root, filename)

    if not os.path.exists(path):
        download_file(ontology_url, path)

    # label
    filename = os.path.basename(label_url)
    path = os.path.join(label_root, filename)

    if not os.path.exists(path):
        download_file(label_url, path)

    # csv
    for csv_url in [balanced_train_csv_url, unbalanced_train_csv_url, eval_csv_url]:
        filename = os.path.basename(csv_url)
        path = os.path.join(csv_root, filename)

        if not os.path.exists(path):
            download_file(csv_url, path)


if __name__ == "__main__":
    main()
