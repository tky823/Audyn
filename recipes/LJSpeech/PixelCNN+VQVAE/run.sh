#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""
pixelcnn_checkpoint=""
vqvae_checkpoint=""

exp_dir="./exp"

ljspeech_url="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
data_root="../data"
dump_root="dump"

system="defaults"
preprocess="defaults"
data="vqvae"
train=""
test="pixelcnn+vqvae"
model=""
optimizer=""
lr_scheduler=""
criterion=""

n_validation=500
n_test=500

. ../../_common/parse_options.sh || exit 1;

set +u

# path to local scripts
export PYTHONPATH="./:${PYTHONPATH}"

set -u

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1"

    (
        . ../_common/download.sh \
        --ljspeech-url "${ljspeech_url}" \
        --data-root "${data_root}"
    )
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Preprocessing"

    (
        . ./preprocess.sh \
        --stage 1 \
        --stop-stage 2 \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --n-validation ${n_validation} \
        --n-test ${n_test}
    )
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Training of VQ-VAE"

    (
        . ./train_vqvae.sh \
        --tag "${tag}" \
        --continue-from "${continue_from}" \
        --dump-root "${dump_root}" \
        --exp-dir "${exp_dir}" \
        --system "${system}" \
        --data "${data}" \
        --train "${train}" \
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

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Training of PixelCNN"

    (
        . ./train_pixelcnn.sh \
        --tag "${tag}" \
        --continue-from "${continue_from}" \
        --exp-dir "${exp_dir}" \
        --data-root "${data_root}" \
        --system "${system}" \
        --data "${data}" \
        --train "${train}" \
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
        --exp-dir "${exp_dir}" \
        --pixelcnn-checkpoint "${pixelcnn_checkpoint}" \
        --vqvae-checkpoint "${vqvae_checkpoint}" \
        --data-root "${data_root}" \
        --system "${system}" \
        --data "${data}" \
        --test "${test}" \
        --model "${model}"
    )
fi
