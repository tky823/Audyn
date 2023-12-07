#!/bin/bash

dump_root="./dump"
exp_dir="./exp"

tag=""
continue_from=""

system="defaults"
preprocess="defaults"
data="wavenet"
train="wavenet"
model="wavenet"
optimizer="adam"
lr_scheduler="wavenet"
criterion="cross_entropy"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

is_distributed=$(
    python ../../_common/is_distributed.py \
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}"
)

if [ "${is_distributed}" = "true" ]; then
    nproc_per_node=$(
        python -c "import torch; print(torch.cuda.device_count())"
    )
    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=${nproc_per_node}"
else
    cmd="python"
fi

dump_format=$(
    python ../../_common/detect_dump_format.py \
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}"
)

if [ "${dump_format}" = "webdataset" ]; then
    train_feature_dir="${feature_dir}/train"
    validation_feature_dir="${feature_dir}/validation"
elif [ "${dump_format}" = "torch" ]; then
    train_feature_dir="${feature_dir}"
    validation_feature_dir="${feature_dir}"
else
    echo "Invalid format ${dump_format} is detected."
    exit 1;
fi

${cmd} ./local/train.py \
--config-dir "./conf" \
hydra.run.dir="${exp_dir}/${tag}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
data="${data}" \
train="${train}" \
model="${model}" \
optimizer="${optimizer}" \
lr_scheduler="${lr_scheduler}" \
criterion="${criterion}" \
train.dataset.train.list_path="${list_dir}/train.txt" \
train.dataset.train.feature_dir="${train_feature_dir}" \
train.dataset.validation.list_path="${list_dir}/validation.txt" \
train.dataset.validation.feature_dir="${validation_feature_dir}" \
train.resume.continue_from="${continue_from}" \
train.output.exp_dir="${exp_dir}/${tag}" \
train.output.tensorboard_dir="tensorboard/${tag}"
