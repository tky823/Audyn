#!/bin/bash

set -eu
set -o pipefail

data_root="../data"
log_root="./log"

. ../../_common/parse_options.sh || exit 1;

audioset_root="${data_root}/AudioSet"
audioset_ontology_root="${audioset_root}/ontology"
audioset_label_root="${audioset_root}/label"
audioset_csv_root="${audioset_root}/csv"

python ../_common/local/download_metadata.py \
--config-dir "../_common/conf" \
hydra.run.dir="${log_root}/$(date +"%Y%m%d-%H%M%S")" \
preprocess="${preprocess}" \
preprocess.ontology_root="${audioset_ontology_root}" \
preprocess.label_root="${audioset_label_root}" \
preprocess.csv_root="${audioset_csv_root}"
