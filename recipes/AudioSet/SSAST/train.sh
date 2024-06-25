#!/bin/bash

dump_root="./dump"
exp_root="./exp"
tensorboard_root="./tensorboard"

tag=""
continue_from=""

train_name="full_train"
validation_name="full_validation"

dump_format="webdataset"

system="defaults"
preprocess="audioset"
data="ssast"
train="ssast"
model="multitask_ssast_patch_mask400"
optimizer="adam"
lr_scheduler="ssast"
criterion="ssast"

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
    audyn-parse-run-command
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}"
)

${cmd} ./local/train.py \
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
train.dataset.train.list_path="${list_dir}/${train_name}.txt" \
train.dataset.train.feature_dir="${feature_dir}/${train_name}" \
train.dataset.validation.list_path="${list_dir}/${validation_name}.txt" \
train.dataset.validation.feature_dir="${feature_dir}/${validation_name}" \
train.resume.continue_from="${continue_from}" \
train.output.exp_dir="${exp_dir}" \
train.output.tensorboard_dir="${tensorboard_dir}"
