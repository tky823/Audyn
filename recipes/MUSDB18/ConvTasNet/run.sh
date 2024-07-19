#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""

exp_root="./exp"
tensorboard_root="./tensorboard"

musdb18_url="https://zenodo.org/records/1117372/files/musdb18.zip"
data_root="../data"
dump_root="dump"

dump_format="torch"

. ../../_common/parse_options.sh || exit 1;

musdb18_root="${data_root}/MUSDB18"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1"

    (
        . ../_common/download.sh \
        --musdb18-url "${musdb18_url}" \
        --data-root "${data_root}"
    )
fi
