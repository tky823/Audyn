#!/bin/bash

set -eu
set -o pipefail

vctk_url="https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
data_root="../data"

. ../../_common/parse_options.sh || exit 1;

vctk_filename=$(basename "${vctk_url}")
vctk_dirname=$(echo ${vctk_filename} | sed "s/\.zip//")

mkdir -p "${data_root}"

if [ ! -e "${data_root}/${vctk_filename}" ]; then
    wget "${vctk_url}" -P "${data_root}"
else
    echo "${data_root}/${vctk_filename} already exists."
fi

if [ ! -d "${data_root}/${vctk_dirname}" ]; then
    (
        cd "${data_root}"
        mkdir -p "${vctk_dirname}"
        unzip "${vctk_filename}" -d "${vctk_dirname}"
        cd -
    )
else
    echo "${data_root}/${vctk_dirname} has been already unzipped."
fi
