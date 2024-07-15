#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""

exp_root="./exp"
tensorboard_root="./tensorboard"

train_name="train-clean-100"
validation_name="dev-clean"
test_name="test-clean"

libritts_url="https://www.openslr.org/resources/60"
data_root="../data"
dump_root="dump"

dump_format="torch"

system="defaults"
preprocess="defaults"
data="hifigan"
train="hifigan"
model="hifigan_v1"
optimizer="hifigan"
lr_scheduler="hifigan"
criterion="hifigan"

min_train=10
ratio_validation=0.05
ratio_test=0.1

. ../../_common/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1"

    (
        . ../_common/download.sh \
        --libritts-url "${libritts_url}" \
        --data-root "${data_root}" \
        --train-name "${train_name}" \
        --validation-name "${validation_name}" \
        --test-name "${test_name}"
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
        --dump-format "${dump_format}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --min-train ${min_train} \
        --ratio-validation ${ratio_validation} \
        --ratio-test ${ratio_test}
    )
fi
