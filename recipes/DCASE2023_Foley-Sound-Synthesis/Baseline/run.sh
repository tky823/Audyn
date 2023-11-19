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

official_data_root="../data"
urbansound8k_data_root="../../UrbanSound8k/data"
dump_root="./dump"

system="defaults"
preprocess="baseline"
data=""
train=""
test="pixelcnn+vqvae"
model=""
optimizer=""
lr_scheduler=""
criterion=""

n_validation=10

. ../../_common/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1"
    echo "Please place dataset under ${data_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Preprocessing of official development dataset"

    (
        . ./preprocess_official_dataset.sh \
        --stage 1 \
        --stop-stage 2 \
        --data-root "${official_data_root}" \
        --dump-root "${dump_root}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --n-validation ${n_validation}
    )
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Preprocessing of UrbanSound8k"

    (
        . ./preprocess_urbansound8k.sh \
        --stage 1 \
        --stop-stage 2 \
        --data-root "${urbansound8k_data_root}" \
        --dump-root "${dump_root}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --n-validation ${n_validation}
    )
fi
