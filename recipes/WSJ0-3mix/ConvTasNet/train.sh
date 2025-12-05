#!/bin/bash

dump_root="./dump"
exp_root="./exp"
tensorboard_root="./tensorboard"

tag=""
continue_from=""

dump_format="torch"

system="default"
preprocess="default"
data="wsj0-3mix_8k"
train="wsj0-mix"
model="conv-tasnet"
optimizer="adam"
lr_scheduler="conv-tasnet"
criterion="neg-sisdr"

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
train.dataset.validation.list_path="${list_dir}/validation.txt" \
train.dataset.validation.feature_dir="${feature_dir}/validation" \
train.resume.continue_from="${continue_from}" \
train.output.exp_dir="${exp_dir}" \
train.output.tensorboard_dir="${tensorboard_dir}"
