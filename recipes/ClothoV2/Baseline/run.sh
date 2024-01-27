#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""

exp_dir="./exp"

clotho_url="https://zenodo.org/records/4783391/files"
data_root="../data"
dump_root="dump"

dump_format="torch"

. ../../_common/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1"

    (
        . ../_common/download.sh \
        --clotho-url "${clotho_url}" \
        --data-root "${data_root}"
    )
fi
