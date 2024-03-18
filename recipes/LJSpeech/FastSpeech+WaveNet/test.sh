#!/bin/bash

dump_root="./dump"
exp_root="./exp"

tag=""
text_to_feat_checkpoint=""
feat_to_wave_checkpoint=""

dump_format="torch"

system="defaults"
preprocess="defaults"
data="ljspeech_text-to-wave"
test="fastspeech+wavenet"
model="fastspeech+wavenet"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

python ./local/test.py \
--config-dir "./conf" \
hydra.run.dir="${exp_root}/${tag}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
preprocess="${preprocess}" \
data="${data}" \
test="${test}" \
model="${model}" \
preprocess.dump_format="${dump_format}" \
test.dataset.test.list_path="${list_dir}/test.txt" \
test.dataset.test.feature_dir="${feature_dir}/test" \
test.checkpoint.text_to_feat="${text_to_feat_checkpoint}" \
test.checkpoint.feat_to_wave="${feat_to_wave_checkpoint}" \
test.output.exp_dir="${exp_root}/${tag}"
