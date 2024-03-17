#!/bin/bash

data_root="../data"
exp_root="./exp"

tag=""
continue_from=""

dump_format="torch"

system="defaults"
data="vqvae"
train="vqvae"
test="pixelcnn+vqvae"
model="vqvae"
optimizer="vqvae"
lr_scheduler="vqvae"
criterion="vqvae_melspectrogram"

. ../../_common/parse_options.sh || exit 1;

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

cmd=$(
    python ../../_common/parse_run_command.py \
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}"
)

${cmd} ./local/train_vqvae.py \
--config-dir "./conf" \
hydra.run.dir="${exp_root}/${tag}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
preprocess="${preprocess}" \
data="${data}" \
train="${train}" \
test="${test}" \
model="${model}" \
optimizer="${optimizer}" \
lr_scheduler="${lr_scheduler}" \
criterion="${criterion}" \
preprocess.dump_format="${dump_format}" \
train.dataset.train.root="${data_root}" \
train.resume.continue_from="${continue_from}" \
train.output.exp_dir="${exp_root}/${tag}" \
train.output.tensorboard_dir="tensorboard/${tag}/vqvae"
