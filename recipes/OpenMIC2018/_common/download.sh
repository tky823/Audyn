#!/bin/bash

set -eu
set -o pipefail

data_root="../data"

. ../../_common/parse_options.sh || exit 1;

openmic2018_root="${data_root}/openmic-2018"

audyn-download-openmic2018 \
root="${data_root}" \
openmic2018_root="${openmic2018_root}"
