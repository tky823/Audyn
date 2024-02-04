#!/bin/bash

dump_root="./dump"
exp_dir="./exp"

tag=""
continue_from=""

dump_format="torch"

system="defaults"
preprocess="clotho-v2"
data="clotho-v2"
test="clap"
model="clap"
metrics="clap"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"

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

${cmd} ./local/test.py \
--config-dir "./conf" \
hydra.run.dir="${exp_dir}/${tag}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
preprocess="${preprocess}" \
data="${data}" \
test="${test}" \
model="${model}" \
preprocess.dump_format="${dump_format}" \
test.dataset.test.list_path="${list_dir}/test.txt" \
test.dataset.test.feature_dir="${save_dir}" \
test.output.exp_dir="${exp_dir}/${tag}"
