#!/bin/bash

dump_root="./dump"
exp_dir="./exp"

tag=""
continue_from=""
hifigan_checkpoint=""

dump_format="torch"

system="defaults"
data="baseline"
train="vqvae"
model="vqvae"
optimizer="vqvae_ema"
lr_scheduler="none"
criterion="vqvae_melspectrogram"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}/pixelsnail+vqvae"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

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
hydra.run.dir="${exp_dir}/${tag}/log/$(date +"%Y%m%d-%H%M%S")" \
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
++train.pretrained_feat_to_wave.path="${hifigan_checkpoint}" \
train.output.exp_dir="${exp_dir}/${tag}" \
train.output.tensorboard_dir="tensorboard/${tag}/vqvae"
