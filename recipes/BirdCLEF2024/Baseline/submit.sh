#!/bin/bash

dump_root="./dump"
exp_root="./exp"

tag=""
submission_path=""

dump_format="birdclef2024"

system="defaults"
preprocess="birdclef2024"
data="birdclef2024"
train="birdclef2024baseline"
test="birdclef2024baseline"
model="birdclef2024baseline"

. ../../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"

if [ -z "${tag}" ]; then
    echo "Set tag in submit.sh."
    exit 1;
fi

test_list_path="${list_dir}/test.txt"

exp_dir="${exp_root}/${tag}"
test_inference_dir="${exp_dir}/inference"

python local/submit.py \
--config-dir "./conf" \
hydra.run.dir="${exp_dir}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
preprocess="${preprocess}" \
data="${data}" \
train="${train}" \
test="${test}" \
model="${model}" \
preprocess.list_path="${test_list_path}" \
preprocess.feature_dir="${test_inference_dir}" \
preprocess.submission_path="${submission_path}"
