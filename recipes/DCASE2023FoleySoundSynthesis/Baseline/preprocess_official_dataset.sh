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

n_validation=10

. ../../_common/parse_options.sh || exit 1;

dataset_root="${data_root}/DCASE_2023_Challenge_Task_7_Dataset"
category_list_path="../_common/category.txt"

dump_dir="${dump_root}/${data}/pixelsnail+vqvae"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"
category_path="${dump_root}/${data}/category.pth"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1"
    echo "Split dev set into training/validation"

    tmp_root="/tmp/$(uuidgen)"
    
    mkdir -p "${tmp_root}"
    mkdir -p "${list_dir}"

    for subset in "train" "validation"; do
        : > "${list_dir}/${subset}.txt"
    done

    while read category; do
        all_list_path="${tmp_root}/${category}.txt"

        ls "${dataset_root}/dev/${category}" > "${all_list_path}"

        n_all=$(wc -l < "${all_list_path}")
        n_train=$((${n_all} - ${n_validation}))

        for filename in $(head -n ${n_train} "${all_list_path}"); do
            echo "dev/${category}/${filename/.wav/}" >> "${list_dir}/train.txt"
        done

        for filename in $(tail -n ${n_validation} "${all_list_path}"); do
            echo "dev/${category}/${filename/.wav/}" >> "${list_dir}/validation.txt"
        done
    done < "${category_list_path}"

    rm -r "${tmp_root}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    python ./local/save_categories.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.category_list_path="${category_list_path}" \
    preprocess.category_path="${category_path}"

    for subset in "train" "validation"; do
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
    done
fi
