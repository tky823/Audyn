#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_dir="./log"

preprocess="defaults"
data="wavenet"

n_validation=500
n_test=500

. ../../_common/parse_options.sh || exit 1;

ljspeech_root="${data_root}/LJSpeech-1.1"
csv_path="${ljspeech_root}/metadata.csv"
wav_dir="${ljspeech_root}/wavs"

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Split data into training/validation/test"

    tmp_root="/tmp/$(uuidgen)"

    mkdir -p "${tmp_root}"

    n_all=$(wc -l < "${csv_path}")
    n_train=$((${n_all} - ${n_validation} - ${n_test}))
    head -n ${n_train} "${csv_path}" > "${tmp_root}/train.csv"
    set +e
    tail -n $((${n_validation} + ${n_test})) "${csv_path}" | head -n ${n_validation} > "${tmp_root}/validation.csv"
    set -e
    tail -n ${n_test} "${csv_path}" > "${tmp_root}/test.csv"

    mkdir -p "${list_dir}"

    IFS="|"

    for subset in "train" "validation" "test"; do
        : > "${list_dir}/${subset}.txt"
        while read filename text normalized_text; do
            echo "${filename}" >> "${list_dir}/${subset}.txt"
        done < "${tmp_root}/${subset}.csv"
    done

    unset IFS

    rm -r "${tmp_root}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    dump_format=$(
        python ../../_common/detect_dump_format.py \
        --config-dir "./conf" \
        hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}"
    )

    for subset in "train" "validation" "test"; do
        if [ "${dump_format}" = "webdataset" ]; then
            subset_feature_dir="${feature_dir}/${subset}"
        elif [ "${dump_format}" = "torch" ]; then
            subset_feature_dir="${feature_dir}"
        else
            echo "Invalid format ${dump_format} is detected."
            exit 1;
        fi

        python ./local/save_features.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.list_path="${list_dir}/${subset}.txt" \
        preprocess.wav_dir="${wav_dir}" \
        preprocess.feature_dir="${subset_feature_dir}"
    done
fi
