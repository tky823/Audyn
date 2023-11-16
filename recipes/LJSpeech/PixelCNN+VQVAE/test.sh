#!/bin/bash

dump_root="./dump"
exp_dir="./exp"

tag=""
pixelcnn_checkpoint=""
vqvae_checkpoint=""

system="defaults"
data="vqvae"
test="pixelcnn+vqvae"
model="pixelcnn+vqvae"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

python ./local/test.py \
--config-dir "./conf" \
hydra.run.dir="${exp_dir}/${tag}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
data="${data}" \
test="${test}" \
model="${model}" \
test.dataset.test.list_path="${list_dir}/test.txt" \
test.dataset.test.feature_dir="${feature_dir}" \
test.checkpoint.pixelcnn="${pixelcnn_checkpoint}" \
test.checkpoint.vqvae="${vqvae_checkpoint}" \
test.output.exp_dir="${exp_dir}/${tag}"
