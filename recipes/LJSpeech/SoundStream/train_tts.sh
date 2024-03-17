#!/bin/bash

dump_root="./dump"
exp_root="./exp"
tensorboard_root="./tensorboard"

tag=""
continue_from=""
feat_to_wave_checkpoint=""

dump_format="torch"

system="defaults"
preprocess="defaults"
data="soundstream"
train="valle"
model="valle"
optimizer="soundstream"
lr_scheduler="soundstream"
criterion="soundstream"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

save_dir="${exp_root}/${tag}/codebook_indices"

cmd=$(
    python ../../_common/parse_run_command.py \
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}"
)

${cmd} ./local/train_tts.py \
--config-dir "./conf" \
hydra.run.dir="${exp_root}/${tag}/log/$(date +"%Y%m%d-%H%M%S")" \
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
train.dataset.train.feature_dir="${save_dir}/train" \
train.dataset.validation.list_path="${list_dir}/validation.txt" \
train.dataset.validation.feature_dir="${save_dir}/validation" \
train.resume.continue_from="${continue_from}" \
++train.pretrained_feat_to_wave.path="${feat_to_wave_checkpoint}" \
train.output.exp_dir="${exp_root}/${tag}" \
train.output.tensorboard_dir="${tensorboard_root}/${tag}/valle"
