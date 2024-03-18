#!/bin/bash

exp_root="./exp"

tag=""
checkpoint=""

dump_format="torch"

system="defaults"
preprocess="defaults"
data="soundstream"
train="valle"

. ../../_common/parse_options.sh || exit 1;

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

filename="$(basename "${checkpoint}")"

cmd=$(
    python ../../_common/parse_run_command.py \
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}"
)

${cmd} ./local/convert_soundstream.py \
--config-dir "./conf" \
hydra.run.dir="${exp_root}/${tag}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
preprocess="${preprocess}" \
data="${data}" \
train="${train}" \
preprocess.dump_format="${dump_format}" \
train.checkpoint="${checkpoint}" \
train.output.save_path="${exp_root}/${tag}/model/soundstream_first_stage_decoder/${filename}"
