#!/bin/bash

set -eu
set -o pipefail

clotho_url="https://zenodo.org/records/4783391/files"
data_root="../data"

. ../../_common/parse_options.sh || exit 1;

mkdir -p "${data_root}"

for subset in "development" "validation" "evaluation"; do
    audio_url="${clotho_url}/clotho_audio_${subset}.7z"
    captions_url="${clotho_url}/clotho_captions_${subset}.csv"
    metadata_url="${clotho_url}/clotho_metadata_${subset}.csv"
    audio_filename="$(basename "${audio_url}")"
    captions_filename="$(basename "${captions_url}")"
    metadata_filename="$(basename "${metadata_url}")"
    audio_path="${data_root}/${audio_filename}"
    captions_path="${data_root}/${captions_filename}"
    metadata_path="${data_root}/${metadata_filename}"
    audio_dirname="${data_root}/${subset}"

    if [ -e "${audio_path}" ]; then
        echo "${audio_path} already exists."
    else
        wget "${audio_url}" -P "${data_root}"
    fi

    if [ -d "${audio_dirname}" ]; then
        echo "${audio_dirname} already exists."
    else
        7z x "${audio_path}" -o"${data_root}"
    fi

    if [ -e "${captions_path}" ]; then
        echo "${captions_path} already exists."
    else
        wget "${captions_url}" -P "${data_root}"
    fi

    if [ -e "${metadata_path}" ]; then
        echo "${metadata_path} already exists."
    else
        wget "${metadata_url}" -P "${data_root}"
    fi
done
