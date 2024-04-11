#!/bin/bash

set -eu
set -o pipefail

audioset_balanced_train_csv_url="http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
audioset_unbalanced_train_csv_url="http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"
audioset_eval_csv_url="http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
data_root="../data"

preprocess="audioset"

. ../../_common/parse_options.sh || exit 1;

audioset_root="${data_root}/AudioSet"
audioset_csv_root="${audioset_root}/csv"
audioset_jsonl_root="${audioset_root}/jsonl"
audioset_m4a_root="${audioset_root}/m4a"

for csv_url in "${audioset_balanced_train_csv_url}" "${audioset_unbalanced_train_csv_url}" "${audioset_eval_csv_url}"; do
    csv_filename="$(basename "${csv_url}")"
    jsonl_filename="${csv_filename/.csv/.jsonl}"
    csv_path="${audioset_csv_root}/${csv_filename}"
    jsonl_path="${audioset_jsonl_root}/${jsonl_filename}"
    download_dir="${audioset_m4a_root}/${csv_filename/.csv/}"

    python local/download_audio.py \
    --config-dir "../_common/conf" \
    hydra.run.dir="log/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    preprocess.csv_path="${csv_path}" \
    preprocess.jsonl_path="${jsonl_path}" \
    preprocess.download_dir="${download_dir}"
done

full_jsonl_path="${audioset_jsonl_root}/full_train.jsonl"
:> "${full_jsonl_path}"

for csv_url in "${audioset_balanced_train_csv_url}" "${audioset_unbalanced_train_csv_url}"; do
    csv_filename="$(basename "${csv_url}")"
    jsonl_filename="${csv_filename/.csv/.jsonl}"
    jsonl_path="${audioset_jsonl_root}/${jsonl_filename}"

    cat "${jsonl_path}" >> "${full_jsonl_path}"
done

full_jsonl_path="${audioset_jsonl_root}/full_validation.jsonl"
:> "${full_jsonl_path}"

csv_filename="$(basename "${audioset_eval_csv_url}")"
jsonl_filename="${csv_filename/.csv/.jsonl}"
jsonl_path="${audioset_jsonl_root}/${jsonl_filename}"

cat "${jsonl_path}" >> "${full_jsonl_path}"
