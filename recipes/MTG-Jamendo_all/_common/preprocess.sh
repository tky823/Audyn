#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_dir="./log"

split=0
dump_format="webdataset"

preprocess="mtg-jamendo"
data="mtg-jamendo_all"

. ../../_common/parse_options.sh || exit 1;

mtg_jamendo_root="${data_root}/MTG-Jamendo/audio"

dump_dir="${dump_root}/${data}/split-${split}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Split data into training/validation/test"

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
        preprocess.mtg_jamendo_root="${mtg_jamendo_root}" \
        preprocess.split=${split} \
        preprocess.subset="${subset}"
    done
fi
