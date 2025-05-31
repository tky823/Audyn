#!/bin/bash

set -eu
set -o pipefail

data_root="../data"

preprocess="audioset"

. ../../_common/parse_options.sh || exit 1;

audioset_root="${data_root}/AudioSet"
audioset_csv_root="${audioset_root}/csv"
audioset_jsonl_root="${audioset_root}/jsonl"
audioset_audio_root="${audioset_root}/audio"

for csv_filename in "balanced_train_segments.csv" "unbalanced_train_segments.csv" "eval_segments.csv"; do
    jsonl_filename="${csv_filename/.csv/.jsonl}"
    csv_path="${audioset_csv_root}/${csv_filename}"
    jsonl_path="${audioset_jsonl_root}/${jsonl_filename}"
    download_dir="${audioset_audio_root}/${csv_filename/.csv/}"

    python ../_common/local/download_audio.py \
    --config-dir "../_common/conf" \
    hydra.run.dir="log/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    preprocess.csv_path="${csv_path}" \
    preprocess.jsonl_path="${jsonl_path}" \
    preprocess.download_dir="${download_dir}"
done

full_jsonl_path="${audioset_jsonl_root}/full_train.jsonl"
:> "${full_jsonl_path}"

for csv_filename in "balanced_train_segments.csv" "unbalanced_train_segments.csv"; do
    jsonl_filename="${csv_filename/.csv/.jsonl}"
    jsonl_path="${audioset_jsonl_root}/${jsonl_filename}"

    cat "${jsonl_path}" >> "${full_jsonl_path}"
done

full_jsonl_path="${audioset_jsonl_root}/full_validation.jsonl"
:> "${full_jsonl_path}"

csv_filename="eval_segments.csv"
jsonl_filename="${csv_filename/.csv/.jsonl}"
jsonl_path="${audioset_jsonl_root}/${jsonl_filename}"

cat "${jsonl_path}" >> "${full_jsonl_path}"
