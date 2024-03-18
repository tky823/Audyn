#!/bin/bash

data_root="../data"
exp_root="./exp"

tag=""
pixelcnn_checkpoint=""
vqvae_checkpoint=""

dump_format="torch"

system="defaults"
data="vqvae"
test="pixelcnn+vqvae"
model="pixelcnn+vqvae"

. ../../_common/parse_options.sh || exit 1;

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
test.dataset.test.root="${data_root}" \
test.checkpoint.pixelcnn="${pixelcnn_checkpoint}" \
test.checkpoint.vqvae="${vqvae_checkpoint}" \
test.output.exp_dir="${exp_root}/${tag}"
