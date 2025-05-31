#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

wsj0_2mix_root="../data/wsj0-mix/2speakers/wav8k/min"
dump_root="./dump"
log_root="./log"

dump_format="torch"

preprocess="default"
data="wsj0-2mix_8k"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Split data into training/validation/test"

    mkdir -p "${list_dir}"

    for subset in "train" "validation" "test"; do
        if [ "${subset}" = "train" ]; then
            subset_name="tr"
        elif [ "${subset}" = "validation" ]; then
            subset_name="cv"
        elif [ "${subset}" = "test" ]; then
            subset_name="tt"
        else
            echo "Invalid subset is given."
            exit 1;
        fi

        wav_dir="${wsj0_2mix_root}/${subset_name}/mix"
        : > "${list_dir}/${subset}.txt"

        for filename in $(ls "${wav_dir}"); do
            echo "${filename/.wav/}" >> "${list_dir}/${subset}.txt"
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    for subset in "train" "validation" "test"; do
        if [ "${subset}" = "train" ]; then
            subset_name="tr"
        elif [ "${subset}" = "validation" ]; then
            subset_name="cv"
        elif [ "${subset}" = "test" ]; then
            subset_name="tt"
        else
            echo "Invalid subset is given."
            exit 1;
        fi

        wav_dir="${wsj0_2mix_root}/${subset_name}"

        python ../_common/local/save_features.py \
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
