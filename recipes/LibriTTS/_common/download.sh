#!/bin/bash

set -eu
set -o pipefail

libritts_url="https://www.openslr.org/resources/60"

data_root="../data"

train_name="train-clean-100"
validation_name="dev-clean"
test_name="test-clean"

. ../../_common/parse_options.sh || exit 1;

libritts_root="${data_root}/LibriTTS"

mkdir -p "${libritts_root}"

for subset_name in "${train_name}" "${validation_name}" "${test_name}"; do
    libritts_subset_url="${libritts_url}/${subset_name}.tar.gz"
    filename="${data_root}/${subset_name}.tar.gz"

    if [ ! -e "${filename}" ]; then
        wget "${libritts_subset_url}" -P "${data_root}"
    else
        echo "${filename} already exists."
    fi

    if [ ! -d "${libritts_root}/${subset_name}" ]; then
        (
            cd "${data_root}"
            tar xvzf "${subset_name}.tar.gz"
            cd -
        )
    else
        echo "${libritts_root}/${subset_name} has been already unzipped."
    fi
done
