#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_root="./log"

dump_format="birdclef2024"

preprocess="birdclef2024"
data="birdclef2024"

. ../../_common/parse_options.sh || exit 1;

birdclef2024_dataroot="${data_root}/birdclef-2024"
csv_path="${birdclef2024_dataroot}/train_metadata.csv"
labeled_audio_root="${birdclef2024_dataroot}/train_audio"
unlabeled_audio_root="${birdclef2024_dataroot}/unlabeled_soundscapes"
test_audio_root="${birdclef2024_dataroot}/test_soundscapes"

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Split data into training/validation"

    mkdir -p "${list_dir}"

    for subset in "train" "validation"; do
        list_path="${list_dir}/${subset}.txt"

        python ../_common/local/save_list.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${list_path}" \
        preprocess.csv_path="${csv_path}" \
        preprocess.audio_root="${labeled_audio_root}" \
        preprocess.subset="${subset}"
    done

    for subset in "unlabeled_train" "unlabeled_validation"; do
        list_path="${list_dir}/${subset}.txt"

        python ../_common/local/save_unlabeled_list.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${list_path}" \
        preprocess.audio_root="${unlabeled_audio_root}" \
        preprocess.subset="${subset}"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    for subset in "train" "validation"; do
        subset_list_path="${list_dir}/${subset}.txt"
        subset_feature_dir="${feature_dir}/${subset}"

        python ../_common/local/save_features.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${subset_list_path}" \
        preprocess.feature_dir="${subset_feature_dir}" \
        preprocess.csv_path="${csv_path}" \
        preprocess.audio_root="${labeled_audio_root}" \
        preprocess.subset="${subset}"
    done

    for subset in "unlabeled_train" "unlabeled_validation"; do
        subset_list_path="${list_dir}/${subset}.txt"
        subset_feature_dir="${feature_dir}/${subset}"

        python ../_common/local/save_unlabeled_features.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${subset_list_path}" \
        preprocess.feature_dir="${subset_feature_dir}" \
        preprocess.audio_root="${unlabeled_audio_root}" \
        preprocess.subset="${subset}"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Preprocess stage 3: Save list of test set."

    mkdir -p "${list_dir}"

    subset="test"
    list_path="${list_dir}/${subset}.txt"

    python ../_common/local/save_list.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_path}" \
    preprocess.csv_path="${csv_path}" \
    preprocess.audio_root="${test_audio_root}" \
    preprocess.subset="${subset}"
fi
