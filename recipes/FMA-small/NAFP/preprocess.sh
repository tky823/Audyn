#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_dir="./log"

dump_format="webdataset"

preprocess="fma"
data="fma-small"

. ../../_common/parse_options.sh || exit 1;

fma_type="small"
fma_root="${data_root}/FMA/${fma_type}"

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Split data into training/validation/evaluation"

    mkdir -p "${list_dir}"

    for subset in "train" "validation" "test"; do
        list_path="${list_dir}/${subset}.txt"

        python ./local/save_list.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${list_path}" \
        preprocess.fma_root="${fma_root}" \
        preprocess.type="${fma_type}" \
        preprocess.subset="${subset}"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    for subset in "train" "validation" "test"; do
        list_path="${list_dir}/${subset}.txt"
        subset_feature_dir="${feature_dir}/${subset}"

        mkdir -p "${subset_feature_dir}"

        python ./local/save_features.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${list_path}" \
        preprocess.feature_dir="${subset_feature_dir}" \
        preprocess.fma_root="${fma_root}"
    done
fi
