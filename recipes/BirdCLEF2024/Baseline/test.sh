#!/bin/bash

dump_root="./dump"
exp_root="./exp"

tag=""
checkpoint=""

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
feature_dir="${dump_dir}/feature"

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

test_list_path="${list_dir}/test.txt"

if [ "${dump_format}" = "birdclef2024" ]; then
    test_feature_dir="${data_root}/birdclef-2024"
else
    test_feature_dir="${feature_dir}/test"
fi

exp_dir="${exp_root}/${tag}"

python ./local/test.py \
--config-dir "./conf" \
hydra.run.dir="${exp_dir}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
preprocess="${preprocess}" \
data="${data}" \
train="${train}" \
test="${test}" \
model="${model}" \
test/dataset="${dump_format}" \
preprocess.dump_format="${dump_format}" \
test.dataset.test.list_path="${test_list_path}" \
test.dataset.test.feature_dir="${test_feature_dir}" \
test.checkpoint="${checkpoint}" \
test.output.exp_dir="${exp_root}/${tag}"
