#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=1

tag=""
continue_from=""

exp_root="./exp"
tensorboard_root="./tensorboard"

data_root="../data"
dump_root="dump"

dump_format="custom"

system="default"
preprocess="default"
data="audioset_1"
train="audioset_poincare_embedding_tiny"
model="poincare_embedding"
optimizer="poincare_embedding"
lr_scheduler="poincare_embedding_tiny"
criterion="poincare_negative_sampling"

cd ../../

# temporarily
ln -s ../recipes/_common/

cd -

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Preprocessing"

    (
        . ./preprocess.sh \
        --stage 1 \
        --stop-stage 1 \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --preprocess "${preprocess}" \
        --data "${data}"
    )
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Training"

    (
        . ./train.sh \
        --tag "${tag}" \
        --continue-from "${continue_from}" \
        --exp-root "${exp_root}" \
        --tensorboard-root "${tensorboard_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --train "${train}" \
        --model "${model}" \
        --optimizer "${optimizer}" \
        --lr-scheduler "${lr_scheduler}" \
        --criterion "${criterion}"
    )
fi

rm -r log/
rm -r dump/
rm -r exp/
rm -r tensorboard/

unlink ../../_common
