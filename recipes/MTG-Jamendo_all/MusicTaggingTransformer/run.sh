#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""

exp_root="./exp"
tensorboard_root="./tensorboard"

data_root="../data"
dump_root="dump"

dump_format="webdataset"

system="defaults"
preprocess="mtg-jamendo"
data="mtg-jamendo_all"
train="music-tagging-transformer_linear-probing"
model="music-tagging-transformer"
optimizer="defaults"
lr_scheduler="defaults"
criterion="mtg-jamendo"

. ../../_common/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1"

    (
        . ../_common/download.sh \
        --data-root "${data_root}"
    )
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Preprocess dataset"

    (
        . ../_common/preprocess.sh \
        --stage 1 \
        --stop-stage 2 \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --preprocess "${preprocess}" \
        --data "${data}"
    )
fi
