#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_dir="./log"

dump_format="torch"

preprocess="birdclef2024"
data="birdclef2024"

. ../../_common/parse_options.sh || exit 1;

birdclef2024_dataroot="${data_root}/birdclef-2024"
csv_path="${birdclef2024_dataroot}/train_metadata.csv"
train_audio_root="${birdclef2024_dataroot}/train_audio"
test_audio_root="${birdclef2024_dataroot}/unlabeled_soundscapes"

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Split data into training/validation/test"

    mkdir -p "${list_dir}"

    for subset in "train" "validation"; do
        list_path="${list_dir}/${subset}.txt"

        python ../_common/local/save_list.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${list_path}" \
        preprocess.csv_path="${csv_path}" \
        preprocess.audio_root="${train_audio_root}" \
        preprocess.subset="${subset}"
    done

    subset="test"
    list_path="${list_dir}/${subset}.txt"

    python ../_common/local/save_test_list.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_path}" \
    preprocess.audio_root="${test_audio_root}" \
    preprocess.subset="${subset}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    for subset in "train" "validation"; do
        subset_list_path="${list_dir}/${subset}.txt"
        subset_feature_dir="${feature_dir}/${subset}"

        python ../_common/local/save_features.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${subset_list_path}" \
        preprocess.feature_dir="${subset_feature_dir}" \
        preprocess.csv_path="${csv_path}" \
        preprocess.audio_root="${train_audio_root}" \
        preprocess.subset="${subset}"
    done
fi
