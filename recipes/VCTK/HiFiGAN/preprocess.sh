#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_dir="./log"

dump_format="torch"

preprocess="default"
data="hifigan_vctk"

n_validation=20
n_test=20

. ../../_common/parse_options.sh || exit 1;

vctk_root="${data_root}/VCTK-Corpus-0.92"
vctk_speakers_path="${vctk_root}/speaker-info.txt"
text_dir="${vctk_root}/txt"
wav_dir="${vctk_root}/wav48_silence_trimmed"

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Split data into training/validation/test"

    mkdir -p "${list_dir}"

    : > "${list_dir}/train.txt"
    : > "${list_dir}/validation.txt"
    : > "${list_dir}/test.txt"

    column_id=0

    while read spk age gender accents region; do
        if [ ${column_id} -eq 0 ]; then
            column_id=$((column_id + 1))
            continue
        fi

        if [ ! -d "${wav_dir}/${spk}/" ] || [ ! -d "${text_dir}/${spk}/" ]; then
            # At least, p315 is missing
            column_id=$((column_id + 1))
            continue
        fi

        all_list_path=/tmp/$(uuidgen).txt

        find "${wav_dir}/${spk}" -name "*_mic2.flac" | \
        sort | \
        sed "s/\.flac//" | \
        awk -F "/" -v spk=${spk} '{printf "%s/%s\n", spk, $NF}' \
        > "${all_list_path}"

        n_all=$(wc -l < ${all_list_path})
        n_train=$((${n_all} - ${n_validation} - ${n_test}))

        if [ ${n_all} -eq 0 ]; then
            # At least, p280 is missing.
            column_id=$((column_id + 1))
            continue
        fi

        cat "${all_list_path}" | \
        head -n ${n_train} \
        >> "${list_dir}/train.txt"

        set +e
        cat "${all_list_path}" | \
        tail -n $((${n_validation} + ${n_test})) | head -n ${n_validation} \
        >> "${list_dir}/validation.txt"
        set -e

        cat "${all_list_path}" | \
        tail -n ${n_test} \
        >> "${list_dir}/test.txt"

        rm "${all_list_path}"
        column_id=$((column_id + 1))
    done < "${vctk_speakers_path}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    for subset in "train" "validation" "test"; do
        python ./local/save_features.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${list_dir}/${subset}.txt" \
        preprocess.wav_dir="${wav_dir}" \
        preprocess.feature_dir="${feature_dir}/${subset}"
    done
fi
