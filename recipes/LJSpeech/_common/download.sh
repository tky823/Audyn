#!/bin/bash

set -eu
set -o pipefail

ljspeech_url="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
data_root="../.data"

. ../../_common/parse_options.sh || exit 1;

ljspeech_filename=$(basename "${ljspeech_url}")
ljspeech_dirname=$(echo ${ljspeech_filename} | sed "s/\.tar\.bz2//")

mkdir -p "${data_root}"

if [ ! -e "${data_root}/${ljspeech_dirname}" ]; then
    wget "${ljspeech_url}" -P "${data_root}"
else
    echo "${data_root}/${ljspeech_dirname} already exists."
fi

if [ ! -d "${data_root}/${ljspeech_dirname}" ]; then
    (
        cd "${data_root}"
        tar jxvf "${ljspeech_filename}"
        cd -
    )
else
    echo "${data_root}/${ljspeech_dirname} has been already unzipped."
fi
