#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_root="./log"

dump_format="torch"

preprocess="baseline"
data="baseline"

n_test=""

. ../../_common/parse_options.sh || exit 1;

dataset_root="${data_root}/DCASE_2023_Challenge_Task_7_Dataset"
category_list_path="../_common/category.txt"

dump_dir="${dump_root}/${data}/test"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"
category_path="${dump_root}/${data}/category.pth"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1"
    echo "Set test IDs"

    tmp_root="/tmp/$(uuidgen)"
    
    mkdir -p "${tmp_root}"
    mkdir -p "${list_dir}"

    subset="test"
    : > "${list_dir}/${subset}.txt"

    while read category; do
        all_list_path="${tmp_root}/${category}.txt"

        if [ -z ${n_test} ]; then
            ls "${dataset_root}/eval/${category}" > "${all_list_path}"

            for filename in $(cat "${all_list_path}"); do
                echo "eval/${category}/${filename/.wav/}" >> "${list_dir}/test.txt"
            done
        else
            echo "Custom number of utterances for test is not supported."

            exit 1;
        fi
    done < "${category_list_path}"

    rm -r "${tmp_root}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    subset="test"

    python ./local/save_official_features.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_dir}/${subset}.txt" \
    preprocess.wav_dir="${dataset_root}" \
    preprocess.feature_dir="${feature_dir}/${subset}" \
    preprocess.category_path="${category_path}"
fi
