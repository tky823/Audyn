#!/bin/bash

set -eu
set -o pipefail

data_root="../data"

. ../../_common/parse_options.sh || exit 1;

audyn-download-ljspeech \
root="${data_root}"
