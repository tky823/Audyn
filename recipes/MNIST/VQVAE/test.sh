#!/bin/bash

exp_dir="./exp"

tag=""
pixelcnn_checkpoint=""
vqvae_checkpoint=""

data_root="../data"

system="defaults"
data="vqvae"
test="pixelcnn+vqvae"
model="pixelcnn+vqvae"

. ../../_common/parse_options.sh || exit 1;

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
test.dataset.test.root="${data_root}" \
test.checkpoint.pixelcnn="${pixelcnn_checkpoint}" \
test.checkpoint.vqvae="${vqvae_checkpoint}" \
test.output.exp_dir="${exp_dir}/${tag}"
