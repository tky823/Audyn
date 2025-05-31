#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_root="./log"

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

    full_list_path="${list_dir}/full_train.txt"
    :> "${full_list_path}"

    subset_name="balanced_train_segments"
    list_path="${list_dir}/${subset_name}.txt"
    :> "${list_path}"

    for path in $(ls "${audioset_m4a_root}/${subset_name}"/*/*.m4a); do
        filename=$(basename "${path}")
        ytid=${filename/.m4a/}
        echo "${subset_name}/${ytid}" >> "${list_path}"
    done

    cat "${list_path}" >> "${full_list_path}"

    subset_name="unbalanced_train_segments"
    list_path="${list_dir}/${subset_name}.txt"
    :> "${list_path}"

    for path in $(ls "${audioset_m4a_root}/${subset_name}"/*/*.m4a); do
        filename=$(basename "${path}")
        ytid=${filename/.m4a/}
        echo "${subset_name}/${ytid}" >> "${list_path}"
    done

    cat "${list_path}" >> "${full_list_path}"

    full_list_path="${list_dir}/full_validation.txt"
    :> "${full_list_path}"

    subset_name="eval_segments"
    list_path="${list_dir}/${subset_name}.txt"
    :> "${list_path}"

    for path in $(ls "${audioset_m4a_root}/${subset_name}"/*/*.m4a); do
        filename=$(basename "${path}")
        ytid=${filename/.m4a/}
        echo "${subset_name}/${ytid}" >> "${list_path}"
    done

    cat "${list_path}" >> "${full_list_path}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    for subset_name in "balanced_train_segments" "unbalanced_train_segments" "eval_segments" "full_train" "full_validation"; do
        jsonl_filename="${subset_name}.jsonl"
        list_path="${list_dir}/${subset_name}.txt"
        subset_feature_dir="${feature_dir}/${subset_name}"
        jsonl_path="${audioset_jsonl_root}/${jsonl_filename}"
        download_dir="${audioset_m4a_root}/${jsonl_filename/.jsonl/}"

        python ./local/save_features.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.list_path="${list_path}" \
        preprocess.feature_dir="${subset_feature_dir}" \
        preprocess.jsonl_path="${jsonl_path}"
    done
fi
