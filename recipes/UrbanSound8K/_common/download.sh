#!/bin/bash

set -eu
set -o pipefail

urbansound8k_url="https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
data_root="../data"

. ../../_common/parse_options.sh || exit 1;

urbansound8k_filename=$(basename "${urbansound8k_url}")
urbansound8k_dirname="${urbansound8k_filename/.tar.gz/}"

mkdir -p "${data_root}"

if [ ! -e "${data_root}/${urbansound8k_dirname}" ]; then
    wget "${urbansound8k_url}" -P "${data_root}"
else
    echo "${data_root}/${urbansound8k_dirname} already exists."
fi

if [ ! -d "${data_root}/${urbansound8k_dirname}" ]; then
    (
        cd "${data_root}"
        tar xvzf "${urbansound8k_filename}"
        cd -
    )
else
    echo "${data_root}/${urbansound8k_dirname} has been already unzipped."
fi
