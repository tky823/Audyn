#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_root="./log"

dump_format="torch"

preprocess="default"
data="hifigan_libritts-clean"

train_name="train-clean-100"
validation_name="dev-clean"
test_name="test-clean"

min_train=10
ratio_validation=0.05
ratio_test=0.1

. ../../_common/parse_options.sh || exit 1;

libritts_root="${data_root}/LibriTTS"
libritts_speakers_path="${libritts_root}/speakers.tsv"

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Split data into training/validation/test"

    # min_train, min_validation=1, and min_test=1 are assumed.
    min_all=$((${min_train} + 2))

    mkdir -p "${list_dir}"

    : > "${list_dir}/train.txt"
    : > "${list_dir}/validation.txt"
    : > "${list_dir}/test.txt"

    column_id=0

    while read speaker gender subset name; do
        if [ ${column_id} -eq 0 ]; then
            column_id=$((column_id + 1))
            continue
        fi

        included_subset_name=""

        for subset in "${train_name}" "${validation_name}" "${test_name}"; do
            speaker_dir="${libritts_root}/${subset}/${speaker}"

            if [ -d "${speaker_dir}" ]; then
                included_subset_name="${subset}"
                break
            fi
        done

        if [ -z "${included_subset_name}" ]; then
            continue
        fi

        all_list_path=/tmp/$(uuidgen).txt
        :> "${all_list_path}"

        for chapter in $(ls "${speaker_dir}"); do
            chapter_dir="${speaker_dir}/${chapter}"

            find "${chapter_dir}" -name "*.wav" | \
            sort | \
            sed "s/\.wav//" | \
            awk -F "/" -v subset=${included_subset_name} -v speaker=${speaker} -v chapter=${chapter} \
            '{printf "%s/%s/%s/%s\n", subset, speaker, chapter, $NF}' \
            >> "${all_list_path}"
        done

        n_all=$(wc -l < ${all_list_path})

        if [ ${n_all} -lt ${min_all} ]; then
            continue
        fi

        n_validation=$(python -c "print(max(int(${n_all} * ${ratio_validation}), 1))")
        n_test=$(python -c "print(max(int((${n_all} - ${n_validation}) * ${ratio_test}), 1))")
        n_train=$((${n_all} - ${n_validation} - ${n_test}))

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
    done < "${libritts_speakers_path}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    for subset in "train" "validation" "test"; do
        python ./local/save_features.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${list_dir}/${subset}.txt" \
        preprocess.wav_dir="${libritts_root}" \
        preprocess.feature_dir="${feature_dir}/${subset}"
    done
fi
