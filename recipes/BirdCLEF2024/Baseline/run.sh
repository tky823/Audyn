#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""
checkpoint=""
submission_path=""

exp_root="./exp"
tensorboard_root="./tensorboard"

data_root="../data"
dump_root="dump"

dump_format="birdclef2024"

system="default"
preprocess="birdclef2024"
data="birdclef2024"
train="birdclef2024baseline"
test="birdclef2024baseline"
model="birdclef2024baseline"
optimizer="adam"
lr_scheduler="none"
criterion="birdclef2024"

. ../../_common/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1: Download dataset"

    echo "Please download dataset from official page and place it to ${data_root}."
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Preprocess dataset"

    (
        . ./preprocess.sh \
        --stage 1 \
        --stop-stage 2 \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --preprocess "${preprocess}" \
        --data "${data}"
    )
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Train EfficientNet"

    (
        . ./train.sh \
        --tag "${tag}" \
        --continue-from "${continue_from}" \
        --exp-root "${exp_root}" \
        --tensorboard-root "${tensorboard_root}" \
        --data-root "${data_root}" \
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

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Preprocess test dataset"

    (
        . ./preprocess.sh \
        --stage 3 \
        --stop-stage 4 \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --preprocess "${preprocess}" \
        --data "${data}"
    )
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Infer by EfficientNet"

    (
        . ./test.sh \
        --tag "${tag}" \
        --checkpoint "${checkpoint}" \
        --exp-root "${exp_root}" \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --train "${train}" \
        --test "${test}" \
        --model "${model}"
    )
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Submit estimation."

    (
        . ./submit.sh \
        --tag "${tag}" \
        --submission-path "${submission_path}" \
        --exp-root "${exp_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --train "${train}" \
        --test "${test}" \
        --model "${model}"
    )
fi
