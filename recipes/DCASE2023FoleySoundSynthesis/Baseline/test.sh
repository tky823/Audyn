#!/bin/bash

dump_root="./dump"
exp_root="./exp"

tag=""
pixelcnn_checkpoint=""
vqvae_checkpoint=""
hifigan_checkpoint=""

dump_format="torch"

system="default"
data="baseline"
test="baseline"
model="baseline"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}/test"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

exp_dir="${exp_root}/${tag}"

python ./local/test.py \
--config-dir "./conf" \
hydra.run.dir="${exp_dir}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
preprocess="${preprocess}" \
data="${data}" \
test="${test}" \
model="${model}" \
preprocess.dump_format="${dump_format}" \
test.dataset.test.list_path="${list_dir}/test.txt" \
test.dataset.test.feature_dir="${feature_dir}/test" \
test.checkpoint.pixelsnail="${pixelsnail_checkpoint}" \
test.checkpoint.vqvae="${vqvae_checkpoint}" \
test.checkpoint.hifigan="${hifigan_checkpoint}" \
test.output.exp_dir="${exp_dir}"
