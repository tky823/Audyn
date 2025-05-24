#!/bin/bash

set -eu
set -o pipefail

data_root="../data"

. ../../_common/parse_options.sh || exit 1;

type="small"
fma_root="${data_root}/FMA/small"

mkdir -p "${fma_root}"

audyn-download-fma \
type="${type}" \
root="${data_root}" \
fma_root="${fma_root}"
