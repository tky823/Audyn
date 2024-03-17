#!/bin/bash

dump_root="./dump"
exp_root="./exp"

tag=""
checkpoint=""

dump_format="torch"

system="defaults"
preprocess="defaults"
data="vqvae"
train="prior"
model="vqvae"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

save_dir="${exp_root}/${tag}/prior"

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

for subset in "train" "validation" "test"; do
    list_path="${list_dir}/${subset}.txt"
    subset_feature_dir="${feature_dir}/${subset}"
    subset_save_dir="${save_dir}/${subset}"

    ${cmd} ./local/save_prior.py \
    --config-dir "./conf" \
    hydra.run.dir="${exp_root}/${tag}/log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}" \
    preprocess="${preprocess}" \
    data="${data}" \
    train="${train}" \
    model="${model}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.feature_dir="${subset_save_dir}" \
    train.dataset.list_path="${list_path}" \
    train.dataset.feature_dir="${subset_feature_dir}" \
    train.checkpoint="${checkpoint}"
done
