#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""
checkpoint=""

exp_dir="./exp"

clotho_url="https://zenodo.org/records/4783391/files"
data_root="../data"
dump_root="dump"

dump_format="torch"

system="defaults"
preprocess="clotho-v2"
data="clotho-v2"
train="clap"
test=""
model="clap"
optimizer="clap"
lr_scheduler="clap"
criterion="info_nce"
metrics="clap"

. ../../_common/parse_options.sh || exit 1;

set +u

# path to local scripts
export PYTHONPATH="./:${PYTHONPATH}"

set -u

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1"

    (
        . ../_common/download.sh \
        --clotho-url "${clotho_url}" \
        --data-root "${data_root}"
    )
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Preprocessing"

    (
        . ./preprocess.sh \
        --stage 0 \
        --stop-stage 3 \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}"
    )
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Training CLAP"

    (
        . ./train_clap.sh \
        --tag "${tag}" \
        --continue-from "${continue_from}" \
        --exp-dir "${exp_dir}" \
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

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Save embeddings of text and audio"

    (
        . ./save_embeddings.sh \
        --tag "${tag}" \
        --checkpoint "${checkpoint}" \
        --exp-dir "${exp_dir}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --test "${test}" \
        --model "${model}"
    )
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 5: Test"

    (
        . ./test.sh \
        --tag "${tag}" \
        --exp-dir "${exp_dir}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --test "${test}" \
        --model "${model}" \
        --metrics "${metrics}"
    )
fi
