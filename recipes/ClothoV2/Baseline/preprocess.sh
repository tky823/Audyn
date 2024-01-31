#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_dir="./log"

dump_format="torch"

preprocess="clotho-v2"
data="clotho-v2"

. ../../_common/parse_options.sh || exit 1;

clotho_root="${data_root}/ClothoV2"
text_dir="${data_root}/captions"

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Normalize captions"

    for raw_subset in "development" "validation" "evaluation"; do
        if [ "${raw_subset}" = "development" ]; then
            subset="train"
        elif [ "${raw_subset}" = "evaluation" ]; then
            subset="test"
        else
            subset="${raw_subset}"
        fi

        csv_path="${clotho_root}/clotho_captions_${raw_subset}.csv"

        python ./local/normalize_captions.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.captions_path="${clotho_root}/clotho_captions_${raw_subset}.csv" \
        preprocess.text_dir="${text_dir}/${subset}"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Split data into training/validation/test"

    for raw_subset in "development" "validation" "evaluation"; do
        if [ "${raw_subset}" = "development" ]; then
            subset="train"
        elif [ "${raw_subset}" = "evaluation" ]; then
            subset="test"
        else
            subset="${raw_subset}"
        fi

        csv_path="${clotho_root}/clotho_captions_${raw_subset}.csv"

        python ./local/save_list.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        preprocess.list_path="${list_dir}/${subset}.txt" \
        preprocess.captions_path="${clotho_root}/clotho_captions_${raw_subset}.csv"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Preprocess stage 3: Save features"

    mkdir -p "${list_dir}"

    for raw_subset in "development" "validation" "evaluation"; do
        if [ "${raw_subset}" = "development" ]; then
            subset="train"
        elif [ "${raw_subset}" = "evaluation" ]; then
            subset="test"
        else
            subset="${raw_subset}"
        fi

        python ./local/save_features.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.list_path="${list_dir}/${subset}.txt" \
        preprocess.wav_dir="${clotho_root}/${raw_subset}" \
        preprocess.text_dir="${text_dir}/${subset}" \
        preprocess.feature_dir="${feature_dir}/${subset}"
    done
fi
