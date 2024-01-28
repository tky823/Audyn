#!/bin/bash

set -eu
set -o pipefail

data_root="../data"
log_dir="./log"

preprocess="clotho-v2"

. ../../_common/parse_options.sh || exit 1;

clotho_root="${data_root}/ClothoV2"
captions_path="${clotho_root}/clotho_captions_development.csv"
vocab_path="${data_root}/vocab.txt"

python ./local/create_vocab.py \
--config-dir "./conf" \
hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
preprocess="${preprocess}" \
preprocess.captions_path="${captions_path}" \
preprocess.vocab_path="${vocab_path}"
