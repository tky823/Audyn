#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""

exp_dir="./exp"

system="defaults"
preprocess="defaults"
data="defaults"
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
