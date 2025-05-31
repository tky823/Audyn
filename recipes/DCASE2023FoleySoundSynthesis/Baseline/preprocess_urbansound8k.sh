#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../../UrbanSound8K/data"
dump_root="./dump"
log_root="./log"

dump_format="torch"

preprocess="default"
data="baseline"

n_validation=10

. ../../_common/parse_options.sh || exit 1;

urbansound8k_root="${data_root}/UrbanSound8K"
csv_path="${urbansound8k_root}/metadata/UrbanSound8K.csv"
wav_dir="${urbansound8k_root}/audio"
class_list_path="../../UrbanSound8K/_common/class.txt"

dump_dir="${dump_root}/${data}/hifigan"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Split data into training/validation"

    tmp_root="/tmp/$(uuidgen)"

    mkdir -p "${tmp_root}"
    mkdir -p "${list_dir}"

    while read class_name; do
        :> "${tmp_root}/${class_name}.csv"
    done < "${class_list_path}"

    IFS=","
    row_idx=0

    while read filename freesound_id start end salience fold class_id class_name; do
        if [ ${row_idx} -gt 0 ]; then
            echo "${filename},${freesound_id},${start},${end},${salience},${fold},${class_id},${class_name}" >> "${tmp_root}/${class_name}.csv"
        fi
        row_idx=$((${row_idx} + 1))
    done < "${csv_path}"

    for subset in "train" "validation"; do
        :> "${tmp_root}/${subset}.csv"
    done

    unset IFS

    while read class_name; do
        n_all=$(wc -l < "${tmp_root}/${class_name}.csv")
        n_train=$((${n_all} - ${n_validation}))

        head -n ${n_train} "${tmp_root}/${class_name}.csv" >> "${tmp_root}/train.csv"
        tail -n ${n_validation} "${tmp_root}/${class_name}.csv" >> "${tmp_root}/validation.csv"
    done < "${class_list_path}"

    IFS=","

    for subset in "train" "validation"; do
        : > "${list_dir}/${subset}.txt"
        while read filename freesound_id start end salience fold class_id class_name; do
            # NOTE: class_name is not directly used in this recipe.
            echo "fold${fold}/${filename/.wav/}" >> "${list_dir}/${subset}.txt"
        done < "${tmp_root}/${subset}.csv"
    done

    unset IFS

    rm -r "${tmp_root}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    for subset in "train" "validation"; do
        python ./local/save_urbansound8k_features.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${list_dir}/${subset}.txt" \
        preprocess.wav_dir="${wav_dir}" \
        preprocess.feature_dir="${feature_dir}/${subset}"
    done
fi
