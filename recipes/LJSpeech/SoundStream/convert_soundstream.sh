#!/bin/bash

exp_root="./exp"

tag=""
checkpoint=""

dump_format="torch"

system="default"
preprocess="default"
data="soundstream"
train="valle"

. ../../_common/parse_options.sh || exit 1;

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

exp_dir="${exp_root}/${tag}"
filename="$(basename "${checkpoint}")"

cmd=$(
    audyn-parse-run-command \
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}"
)

${cmd} ./local/convert_soundstream.py \
--config-dir "./conf" \
hydra.run.dir="${exp_dir}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
preprocess="${preprocess}" \
data="${data}" \
train="${train}" \
preprocess.dump_format="${dump_format}" \
train.checkpoint="${checkpoint}" \
train.output.save_path="${exp_dir}/model/soundstream_first_stage_decoder/${filename}"
