#!/bin/bash

exp_dir="./exp"

tag=""
continue_from=""

data_root="../data"

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

list_dir="${exp_dir}/${tag}/list"
feature_dir="${exp_dir}/${tag}/prior"

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

${cmd} ./local/train_pixelcnn.py \
--config-dir "./conf" \
hydra.run.dir="${exp_dir}/${tag}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
data="${data}" \
train="${train}" \
test="${test}" \
model="${model}" \
optimizer="${optimizer}" \
lr_scheduler="${lr_scheduler}" \
criterion="${criterion}" \
train.dataset.train.list_path="${list_dir}/train.txt" \
train.dataset.train.feature_dir="${feature_dir}" \
train.dataset.validation.list_path="${list_dir}/validation.txt" \
train.dataset.validation.feature_dir="${feature_dir}" \
train.resume.continue_from="${continue_from}" \
train.output.exp_dir="${exp_dir}/${tag}" \
train.output.tensorboard_dir="tensorboard/${tag}/pixelcnn"
