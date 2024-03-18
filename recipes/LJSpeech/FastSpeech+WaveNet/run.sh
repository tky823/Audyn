#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
text_to_feat_checkpoint=""
feat_to_wave_checkpoint=""

exp_root="./exp"

dump_root="dump"

dump_format="torch"

system="defaults"
preprocess="defaults"
data="ljspeech_text-to-wave"
test="fastspeech+wavenet"
model="fastspeech+wavenet"

. ../../_common/parse_options.sh || exit 1;

if [ ${stage} -le 1 ]; then
    echo "Run"
    echo "../FastSpeech/run.sh --stage ${stage} --stop-stage ${stop_stage}"
    echo "and"
    echo "../WaveNet/run.sh --stage ${stage} --stop-stage ${stop_stage}"
    echo "instead."

    exit 0
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Test"

    (
        . ./test.sh \
        --tag "${tag}" \
        --text-to-feat-checkpoint "${text_to_feat_checkpoint}" \
        --feat-to-wave-checkpoint "${feat_to_wave_checkpoint}" \
        --exp-root "${exp_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --test "${test}" \
        --model "${model}"
    )
fi
