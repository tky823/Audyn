#!/bin/bash

set -eu
set -o pipefail

dump_root="./dump"
exp_root="./exp"
tensorboard_root="./tensorboard"

tag=""
continue_from=""

dump_format="fma-small_nafp"

system="default"
preprocess="fma-small"
data="fma-small"
train="nafp_fma-small"
model="nafp"
optimizer="nafp"
lr_scheduler="nafp"
criterion="ntxent"

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
train.dataset.train.list_path="${list_dir}/training.txt" \
train.dataset.train.feature_dir="${feature_dir}/training" \
train.dataset.train.background_list_path="${feature_dir}/training_background.txt" \
train.dataset.train.impulse_response_list_path="${feature_dir}/training_impulse-response.txt" \
train.dataset.validation.list_path="${list_dir}/validation.txt" \
train.dataset.validation.feature_dir="${feature_dir}/validation" \
train.dataset.validation.background_list_path="${feature_dir}/validation_background.txt" \
train.dataset.validation.impulse_response_list_path="${feature_dir}/validation_impulse-response.txt" \
train.resume.continue_from="${continue_from}" \
train.output.exp_dir="${exp_dir}" \
train.output.tensorboard_dir="${tensorboard_dir}"
