#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""

exp_root="./exp"
tensorboard_root="./tensorboard"

musdb18_7s_url="https://zenodo.org/api/files/1ff52183-071a-4a59-923f-7a31c4762d43/MUSDB18-7-STEMS.zip"
data_root="../data"
dump_root="dump"

dump_format="torch"

. ../../_common/parse_options.sh || exit 1;

musdb18_7s_root="${data_root}/MUSDB18-7s"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1"

    (
        . ../_common/download.sh \
        --musdb18-7s-url "${musdb18_7s_url}" \
        --data-root "${data_root}"
    )
fi
