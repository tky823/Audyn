#!/bin/bash

data_root="../data"
exp_root="./exp"
tensorboard_root="./tensorboard"

tag=""
continue_from=""

dump_format="torch"

system="defaults"
data="vqvae"
train="pixelcnn"
test="pixelcnn+vqvae"
model="pixelcnn"
optimizer="pixelcnn"
lr_scheduler="pixelcnn"
criterion="pixelcnn"

. ../../_common/parse_options.sh || exit 1;

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

list_dir="${exp_root}/${tag}/list"
feature_dir="${exp_root}/${tag}/prior"

cmd=$(
    python ../../_common/parse_run_command.py \
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}"
)

${cmd} ./local/train_pixelcnn.py \
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
train.dataset.train.list_path="${list_dir}/train.txt" \
train.dataset.train.feature_dir="${feature_dir}" \
train.dataset.validation.list_path="${list_dir}/validation.txt" \
train.dataset.validation.feature_dir="${feature_dir}" \
train.resume.continue_from="${continue_from}" \
train.output.exp_dir="${exp_root}/${tag}" \
train.output.tensorboard_dir="${tensorboard_root}/${tag}/pixelcnn"
