#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_root="./log"

dump_format="musdb18"

preprocess="musdb18"
data="musdb18"

. ../../_common/parse_options.sh || exit 1;

musdb18_7s_root="${data_root}/MUSDB18-7s"

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Split data into training/validation/test"

    mkdir -p "${list_dir}"

    for subset in "train" "validation" "test"; do
        list_path="${list_dir}/${subset}.txt"

        python ../_common/local/save_list.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${list_path}" \
        preprocess.musdb18_root="${musdb18_7s_root}" \
        preprocess.subset="${subset}"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    for subset in "train" "validation" "test"; do
        if [ "${subset}" = "train" ] || [ "${subset}" = "validation" ]; then
            subset_name="train"
        elif [ "${subset}" = "test" ]; then
            subset_name="test"
        else
            echo "Invalid subset is given."
            exit 1;
        fi

        list_path="${list_dir}/${subset}.txt"
        subset_feature_dir="${feature_dir}/${subset}"
        subset_musdb18_dir="${musdb18_7s_root}/${subset_name}"
        relative_dir="$(python -c "import os; print(os.path.relpath('${subset_musdb18_dir}', '${subset_feature_dir}'))")"

        mkdir -p "${subset_feature_dir}"

        while read filename; do
            (
                cd "${subset_feature_dir}"
                ln -s "${relative_dir}/${filename}/"
                cd -
            )
        done < "${list_path}"
    done
fi
