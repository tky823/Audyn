#!/bin/bash

dump_root="./dump"
exp_root="./exp"

tag=""
checkpoint=""

dump_format="torch"

system="defaults"
preprocess="clotho-v2"
data="clotho-v2"
test="save_embeddings"
model="clap"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

exp_dir="${exp_root}/${tag}"
save_dir="${exp_dir}/embeddings"

cmd=$(
    python ../../_common/parse_run_command.py \
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}"
)

for subset in "train" "validation" "test"; do
    subset_save_dir="${save_dir}/${subset}"
    subset_list_path="${list_dir}/${subset}.txt"
    subset_feature_dir="${feature_dir}/${subset}"

    ${cmd} ./local/save_embeddings.py \
    --config-dir "./conf" \
    hydra.run.dir="${exp_dir}/log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}" \
    preprocess="${preprocess}" \
    data="${data}" \
    test="${test}" \
    model="${model}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.feature_dir="${subset_save_dir}" \
    test.dataset.test.list_path="${subset_list_path}" \
    test.dataset.test.feature_dir="${subset_feature_dir}" \
    test.checkpoint="${checkpoint}" \
    test.output.exp_dir="${exp_dir}"
done
