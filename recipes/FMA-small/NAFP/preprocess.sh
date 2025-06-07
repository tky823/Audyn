#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_root="./log"

dump_format="fma-small_nafp"

preprocess="fma"
data="fma-small"

. ../../_common/parse_options.sh || exit 1;

fma_root="${data_root}/neural-audio-fp-dataset"

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Split data into training/validation/evaluation"

    mkdir -p "${list_dir}"

    subset="train"
    list_path="${list_dir}/${subset}.txt"

    python ./local/save_training_list.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_path}" \
    preprocess.fma_root="${fma_root}" \
    preprocess.subset="${subset}"

    list_path="${list_dir}/${subset}_background.txt"

    python ./local/save_training_background_list.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_path}" \
    preprocess.fma_root="${fma_root}" \
    preprocess.subset="${subset}"

    list_path="${list_dir}/${subset}_impulse-response.txt"

    python ./local/save_training_impulse_response_list.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_path}" \
    preprocess.fma_root="${fma_root}" \
    preprocess.subset="${subset}"

    subset="validation"
    list_path="${list_dir}/${subset}.txt"

    python ./local/save_validation_list.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_path}" \
    preprocess.fma_root="${fma_root}" \
    preprocess.subset="${subset}"

    list_path="${list_dir}/${subset}_background.txt"

    python ./local/save_training_background_list.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_path}" \
    preprocess.fma_root="${fma_root}" \
    preprocess.subset="${subset}"

    list_path="${list_dir}/${subset}_impulse-response.txt"

    python ./local/save_training_impulse_response_list.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_path}" \
    preprocess.fma_root="${fma_root}" \
    preprocess.subset="${subset}"

    subset="test"
    list_path="${list_dir}/${subset}_db.txt"

    python ./local/save_evaluation_db_list.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_path}" \
    preprocess.fma_root="${fma_root}" \
    preprocess.subset="${subset}"

    list_path="${list_dir}/${subset}_query.txt"

    python ./local/save_evaluation_query_list.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_path}" \
    preprocess.fma_root="${fma_root}" \
    preprocess.subset="${subset}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    for subset in "train" "validation"; do
        list_path="${list_dir}/${subset}.txt"
        subset_feature_dir="${feature_dir}/${subset}"

        mkdir -p "${subset_feature_dir}"

        python ./local/save_features.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${list_path}" \
        preprocess.feature_dir="${subset_feature_dir}" \
        preprocess.fma_root="${fma_root}" \
        preprocess.subset="${subset}"
    done

    subset="test"
    subset_feature_dir="${feature_dir}/${subset}"

    mkdir -p "${subset_feature_dir}"

    list_path="${list_dir}/${subset}_db.txt"

    python ./local/save_evaluation_db_features.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_path}" \
    preprocess.feature_dir="${subset_feature_dir}" \
    preprocess.fma_root="${fma_root}" \
    preprocess.subset="${subset}"

    list_path="${list_dir}/${subset}_query.txt"

    python ./local/save_evaluation_query_features.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_path}" \
    preprocess.feature_dir="${subset_feature_dir}" \
    preprocess.fma_root="${fma_root}" \
    preprocess.subset="${subset}"
fi
