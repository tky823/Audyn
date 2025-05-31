#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""

exp_root="./exp"
tensorboard_root="./tensorboard"

train_name="full_train"
validation_name="full_validation"

data_root="../data"
dump_root="dump"

dump_format="webdataset"

system="default"
preprocess="hubert"
data="hubert"

. ../../_common/parse_options.sh || exit 1;

audioset_root="${data_root}/AudioSet"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1: Download dataset"

    (
        . ../_common/download_metadata.sh \
        --data-root "${data_root}" \
        --preprocess "${preprocess}"
    )

    (
        . ../_common/download_audio.sh \
        --data-root "${data_root}" \
        --preprocess "${preprocess}"
    )
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Preprocess dataset"

    (
        . ./preprocess.sh \
        --stage 0 \
        --stop-stage 6 \
        --train-name "${train_name}" \
        --validation-name "${validation_name}" \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --preprocess "${preprocess}" \
        --data "${data}"
    )
fi
