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

data_root="../data"
dump_root="./dump"

system="defaults"
preprocess="baseline"
data="vqvae"
train=""
test="pixelcnn+vqvae"
model=""
optimizer=""
lr_scheduler=""
criterion=""

n_validation=10

. ../../_common/parse_options.sh || exit 1;

dataset_root="${data_root}/DCASE_2023_Challenge_Task_7_Dataset"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1"
    echo "Please place dataset under ${data_root}"
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
        --n-validation ${n_validation}
    )
fi
