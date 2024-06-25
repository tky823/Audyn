#!/bin/bash

exp_root="./exp"

tag=""

system="defaults"

. ../_common/parse_options.sh || exit 1;

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

exp_dir="${exp_root}/${tag}"

cmd=$(
    audyn-parse-run-command \
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}"
)

${cmd} ./local/train.py \
--config-dir "./conf" \
hydra.run.dir="${exp_dir}/log/$(date +"%Y%m%d-%H%M%S")"
