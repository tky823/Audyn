#!/bin/bash

set -eu
set -o pipefail

musdb18_url="https://zenodo.org/records/1117372/files/musdb18.zip"
data_root="../data"

. ../../_common/parse_options.sh || exit 1;

musdb18_filename=$(basename "${musdb18_url}")
musdb18_root="${data_root}/MUSDB18"

mkdir -p "${data_root}"

if [ ! -e "${data_root}/${musdb18_filename}" ]; then
    wget "${musdb18_url}" -P "${data_root}"
else
    echo "${data_root}/${musdb18_filename} already exists."
fi

if [ ! -e "${musdb18_root}/train/A Classic Education - NightOwl.stem.mp4" ]; then
    unzip "${data_root}/${musdb18_filename}" -d "${musdb18_root}"
else
    echo "${musdb18_root} already exists."
fi

audyn-decode-musdb18 \
mp4_root="${musdb18_root}" \
wav_root="${musdb18_root}" \
subset="all"
