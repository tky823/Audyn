#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""

ljspeech_url="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
data_root="../data"
dump_root="dump"

system="defaults"
preprocess="ljspeech_text-to-feat"
data="ljspeech_text-to-feat"
train="fastspeech"
model="fastspeech"
optimizer="fastspeech"
lr_scheduler="fastspeech"
criterion="fastspeech"

n_validation=5
n_test=5

. ../../_common/parse_options.sh || exit 1;

ljspeech_root="${data_root}/LJSpeech-1.1"

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
        --stage 0 \
        --stop-stage 4 \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --n-validation ${n_validation} \
        --n-test ${n_test}
    )
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Training"

    (
        . ./train.sh \
        --tag "${tag}" \
        --continue-from "${continue_from}" \
        --dump-root "${dump_root}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --train "${train}" \
        --model "${model}" \
        --optimizer "${optimizer}" \
        --lr_scheduler "${lr_scheduler}" \
        --criterion "${criterion}"
    )
fi
