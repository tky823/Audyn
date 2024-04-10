#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_dir="./log"

dump_format="torch"

preprocess="audioset"
data="audioset"

. ../../_common/parse_options.sh || exit 1;

audioset_root="${data_root}/AudioSet"
audioset_csv_root="${audioset_root}/csv"
audioset_jsonl_root="${audioset_root}/jsonl"
audioset_m4a_root="${audioset_root}/m4a"

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Split data into training/validation"

    mkdir -p "${list_dir}"

    subset="balanced_train"
    subset_name="balanced_train_segments"
    list_path="${list_dir}/${subset}.txt"
    :> "${list_path}"

    for path in $(ls "${audioset_m4a_root}/${subset_name}"/*/*.m4a); do
        filename=$(basename "${path}")
        ytid=${filename/.m4a/}
        echo ${ytid} >> "${list_path}"
    done

    subset="unbalanced_train"
    subset_name="unbalanced_train_segments"
    list_path="${list_dir}/${subset}.txt"
    :> "${list_path}"

    for path in $(ls "${audioset_m4a_root}/${subset_name}"/*/*.m4a); do
        filename=$(basename "${path}")
        ytid=${filename/.m4a/}
        echo ${ytid} >> "${list_path}"
    done

    subset="validation"
    subset_name="eval_segments"
    list_path="${list_dir}/${subset}.txt"
    :> "${list_path}"

    for path in $(ls "${audioset_m4a_root}/${subset_name}"/*/*.m4a); do
        filename=$(basename "${path}")
        ytid=${filename/.m4a/}
        echo ${ytid} >> "${list_path}"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    mkdir -p "${list_dir}"

    for subset in "balanced_train" "unbalanced_train" "validation"; do
        if [ "${subset}" = "balanced_train" ]; then
            jsonl_filename="balanced_train_segments.jsonl"
        
        elif [ "${subset}" = "unbalanced_train" ]; then
            jsonl_filename="unbalanced_train_segments.jsonl"
        else
            jsonl_filename="eval_segments.jsonl"
        fi

        list_path="${list_dir}/${subset}.txt"
        jsonl_path="${audioset_jsonl_root}/${jsonl_filename}"
        download_dir="${audioset_m4a_root}/${jsonl_filename/.jsonl/}"

        python ./local/save_features.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.list_path="${list_path}" \
        preprocess.feature_dir="${feature_dir}/${subset}" \
        preprocess.jsonl_path="${jsonl_path}" \
        preprocess.download_dir="${download_dir}"
    done
fi
