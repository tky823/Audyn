#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""
pixelcnn_checkpoint=""
vqvae_checkpoint=""

exp_root="./exp"
tensorboard_root="./tensorboard"

data_root="../data"

dump_format="torch"

system="default"
preprocess="default"
data="rvqvae"
train=""
test="pixelcnn+vqvae"
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
    echo "Stage 1: Training of RVQVAE"

    (
        . ./train_vqvae.sh \
        --tag "${tag}" \
        --continue-from "${continue_from}" \
        --exp-root "${exp_root}" \
        --tensorboard-root "${tensorboard_root}" \
        --data-root "${data_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --train "${train}" \
        --test "${test}" \
        --model "${model}" \
        --optimizer "${optimizer}" \
        --lr-scheduler "${lr_scheduler}" \
        --criterion "${criterion}"
    )
fi
