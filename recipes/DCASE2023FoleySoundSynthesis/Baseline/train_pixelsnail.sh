#!/bin/bash

dump_root="./dump"
exp_root="./exp"
tensorboard_root="./tensorboard"

tag=""
continue_from=""

dump_format="torch"

system="defaults"
data="baseline"
train="pixelsnail"
model="pixelsnail"
optimizer="pixelsnail"
lr_scheduler="none"
criterion="pixelsnail"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}/pixelsnail+vqvae"
list_dir="${dump_dir}/list"

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

exp_dir="${exp_root}/${tag}"
feature_dir="${exp_dir}/prior"

cmd=$(
    python ../../_common/parse_run_command.py \
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}"
)

${cmd} ./local/train_pixelsnail.py \
--config-dir "./conf" \
hydra.run.dir="${exp_dir}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
preprocess="${preprocess}" \
data="${data}" \
train="${train}" \
model="${model}" \
optimizer="${optimizer}" \
lr_scheduler="${lr_scheduler}" \
criterion="${criterion}" \
preprocess.dump_format="${dump_format}" \
train.dataset.train.list_path="${list_dir}/train.txt" \
train.dataset.train.feature_dir="${feature_dir}/train" \
train.dataset.validation.list_path="${list_dir}/validation.txt" \
train.dataset.validation.feature_dir="${feature_dir}/validation" \
train.resume.continue_from="${continue_from}" \
train.output.exp_dir="${exp_dir}" \
train.output.tensorboard_dir="${tensorboard_root}/${tag}/pixelsnail"
