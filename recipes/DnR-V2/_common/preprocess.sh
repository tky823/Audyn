#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_dir="./log"

dump_format="dnr-v2"

preprocess="dnr-v2"
data="dnr-v2"

. ../../_common/parse_options.sh || exit 1;

dnr_root="${data_root}/DnR-V2"

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Split data into training/validation/evaluation"

    mkdir -p "${list_dir}"

    for subset in "train" "validation" "test"; do
        list_path="${list_dir}/${subset}.txt"

        python ../_common/local/save_list.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.dump_format="${dump_format}" \
        preprocess.list_path="${list_path}" \
        preprocess.dnr_root="${dnr_root}" \
        preprocess.subset="${subset}"
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

        list_path="${list_dir}/${subset}.txt"
        subset_feature_dir="${feature_dir}/${subset}"
        subset_dnr_dir="${dnr_root}/${subset_name}"
        relative_dir="$(python -c "import os; print(os.path.relpath('${subset_dnr_dir}', '${subset_feature_dir}'))")"

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
