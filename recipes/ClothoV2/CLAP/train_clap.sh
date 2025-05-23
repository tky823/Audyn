#!/bin/bash

dump_root="./dump"
exp_root="./exp"
tensorboard_root="./tensorboard"

tag=""
continue_from=""
text_tower_checkpoint=""
audio_tower_checkpoint=""

dump_format="torch"

system="default"
preprocess="clotho-v2"
data="clotho-v2"
train="baseline"
model="clap"
optimizer="clap"
lr_scheduler="clap"
criterion="info_nce"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

exp_dir="${exp_root}/${tag}"
tensorboard_dir="${tensorboard_root}/${tag}"

cmd=$(
    audyn-parse-run-command \
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}"
)

additional_args=""

if [ -n "${text_tower_checkpoint}" ]; then
    additional_args="${additional_args} model.text_tower.path=${text_tower_checkpoint}"
fi

if [ -n "${audio_tower_checkpoint}" ]; then
    additional_args="${additional_args} model.audio_tower.path=${audio_tower_checkpoint}"
fi

if [ -z "${additional_args}" ]; then
    # to avoid error of hydra
    additional_args=" "
fi

${cmd} ./local/train_clap.py \
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
train.output.exp_dir="${exp_dir}/clap" \
train.output.tensorboard_dir="${tensorboard_dir}/clap" \
${additional_args}
