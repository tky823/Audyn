#!/bin/bash

dump_root="./dump"
exp_dir="./exp"

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

save_dir="${exp_dir}/${tag}/embeddings"

cmd=$(
    python ../../_common/parse_run_command.py \
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}"
)

${cmd} ./local/save_embeddings.py \
--config-dir "./conf" \
hydra.run.dir="${exp_dir}/${tag}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
preprocess="${preprocess}" \
data="${data}" \
test="${test}" \
model="${model}" \
preprocess.dump_format="${dump_format}" \
preprocess.feature_dir="${save_dir}" \
test.dataset.test.list_path="${list_dir}/test.txt" \
test.dataset.test.feature_dir="${feature_dir}/test" \
test.checkpoint="${checkpoint}" \
test.output.exp_dir="${exp_dir}/${tag}"