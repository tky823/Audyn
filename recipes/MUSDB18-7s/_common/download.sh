#!/bin/bash

set -eu
set -o pipefail

musdb18_7s_url="https://zenodo.org/api/files/1ff52183-071a-4a59-923f-7a31c4762d43/MUSDB18-7-STEMS.zip"
data_root="../data"

. ../../_common/parse_options.sh || exit 1;

musdb18_7s_filename=$(basename "${musdb18_7s_url}")
musdb18_7s_root="${data_root}/MUSDB18-7s"

subset_names=(train test)

mkdir -p "${data_root}"

if [ ! -e "${data_root}/${musdb18_7s_filename}" ]; then
    wget "${musdb18_7s_url}" -P "${data_root}"
else
    echo "${data_root}/${musdb18_7s_filename} already exists."
fi

if [ ! -e "${musdb18_7s_root}/train/A Classic Education - NightOwl.stem.mp4" ]; then
    unzip "${data_root}/${musdb18_7s_filename}" -d "${musdb18_7s_root}"
else
    echo "${musdb18_7s_root} already exists."
fi

audyn-decode-musdb18 \
mp4_root="${musdb18_7s_root}" \
wav_root="${musdb18_7s_root}" \
subset="all"
