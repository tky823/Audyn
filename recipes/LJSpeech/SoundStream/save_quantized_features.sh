#!/bin/bash

dump_root="./dump"
exp_root="./exp"

tag=""
checkpoint=""

dump_format="torch"

system="defaults"
preprocess="defaults"
data="soundstream"
train="save_quantized_features"
model="soundstream"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

save_dir="${exp_root}/${tag}/codebook_indices"

for subset in "train" "validation" "test"; do
    subset_list_path="${list_dir}/${subset}.txt"
    subset_feature_dir="${feature_dir}/${subset}"
    subset_save_dir="${save_dir}/${subset}"

    python ./local/save_quantized_features.py \
    --config-dir "./conf" \
    hydra.run.dir="${exp_root}/${tag}/log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}" \
    preprocess="${preprocess}" \
    data="${data}" \
    train="${train}" \
    model="${model}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.feature_dir="${subset_save_dir}" \
    train.dataset.list_path="${subset_list_path}" \
    train.dataset.feature_dir="${subset_feature_dir}" \
    train.checkpoint="${checkpoint}" \
    train.output.exp_dir="${exp_root}/${tag}"
done
