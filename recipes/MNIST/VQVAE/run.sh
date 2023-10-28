#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""
vqvae_checkpoint=""

exp_dir="./exp"

data_root="../data"

system="defaults"
preprocess="defaults"
data="vqvae"
train=""
test=""
model=""
optimizer=""
lr_scheduler=""
criterion=""

. ../../_common/parse_options.sh || exit 1;

set +u

# path to local scripts
export PYTHONPATH="./:${PYTHONPATH}"

set -u

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Training of VQ-VAE"

    (
        . ./train_vqvae.sh \
        --tag "${tag}" \
        --continue-from "${continue_from}" \
        --exp-dir "${exp_dir}" \
        --data-root "${data_root}" \
        --system "${system}" \
        --data "${data}" \
        --train "${train}" \
        --test "${test}" \
        --model "${model}" \
        --optimizer "${optimizer}" \
        --lr_scheduler "${lr_scheduler}" \
        --criterion "${criterion}"
    )
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Save prior from VQ-VAE"

    (
        . ./save_prior.sh \
        --tag "${tag}" \
        --exp-dir "${exp_dir}" \
        --checkpoint "${vqvae_checkpoint}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --train "${train}" \
        --model "${model}"
    )
fi
