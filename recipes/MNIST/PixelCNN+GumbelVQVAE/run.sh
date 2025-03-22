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
data="vqvae"
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
    echo "Stage 1: Training of Gumbel-VQVAE"

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
        --lr_scheduler "${lr_scheduler}" \
        --criterion "${criterion}"
    )
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Save prior from Gumbel-VQVAE"

    (
        . ./save_prior.sh \
        --tag "${tag}" \
        --data-root "${data_root}" \
        --dump-format "${dump_format}" \
        --exp-root "${exp_root}" \
        --checkpoint "${vqvae_checkpoint}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --train "${train}" \
        --model "${model}"
    )
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Training of PixelCNN"

    (
        . ./train_pixelcnn.sh \
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
        --lr_scheduler "${lr_scheduler}" \
        --criterion "${criterion}"
    )
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Generate images by PixelCNN + VQVAE"

    (
        . ./test.sh \
        --tag "${tag}" \
        --exp-root "${exp_root}" \
        --pixelcnn-checkpoint "${pixelcnn_checkpoint}" \
        --vqvae-checkpoint "${vqvae_checkpoint}" \
        --data-root "${data_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --test "${test}" \
        --model "${model}"
    )
fi
