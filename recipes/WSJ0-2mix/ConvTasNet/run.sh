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

dump_format="torch"

system="defaults"
preprocess="defaults"
data="wsj0-2mix"
train="wsj0-2mix"
model="conv-tasnet"
optimizer="conv-tasnet"
lr_scheduler="conv-tasnet"
criterion="neg-sisdr"

. ../../_common/parse_options.sh || exit 1;

wsj0_mix_root="${data_root}/wsj0-mix/2speakers/wav8k/min"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1"
    echo "Prepare dataset under ${wsj0_mix_root} on your own."
fi
