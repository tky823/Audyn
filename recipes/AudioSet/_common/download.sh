#!/bin/bash

set -eu
set -o pipefail

audioset_ontology_url="https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
audioset_label_url="http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
audioset_balanced_train_csv_url="http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
audioset_unbalanced_train_csv_url="http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"
audioset_eval_csv_url="http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
data_root="../data"

. ../../_common/parse_options.sh || exit 1;

audioset_root="${data_root}/AudioSet"
audioset_ontology_root="${audioset_root}/ontology"
audioset_label_root="${audioset_root}/label"
audioset_csv_root="${audioset_root}/csv"

mkdir -p "${audioset_ontology_root}"
mkdir -p "${audioset_csv_root}"

ontology_filename="$(basename "${audioset_ontology_url}")"
ontology_path="${audioset_ontology_root}/${ontology_filename}"

if [ -e "${ontology_path}" ]; then
    echo "${ontology_path} already exists."
else
    wget "${audioset_ontology_url}" -P "${audioset_ontology_root}"
fi

label_filename="$(basename "${audioset_label_url}")"
label_path="${audioset_label_root}/${label_filename}"

if [ -e "${label_path}" ]; then
    echo "${label_path} already exists."
else
    wget "${audioset_label_url}" -P "${audioset_label_root}"
fi

for csv_url in "${audioset_balanced_train_csv_url}" "${audioset_unbalanced_train_csv_url}" "${audioset_eval_csv_url}"; do
    csv_filename="$(basename "${csv_url}")"
    csv_path="${audioset_csv_root}/${csv_filename}"

    if [ -e "${csv_path}" ]; then
        echo "${csv_path} already exists."
    else
        wget "${csv_url}" -P "${audioset_csv_root}"
    fi
done
