#!/bin/bash

set -eu
set -o pipefail

data_root="../data"

. ../../_common/parse_options.sh || exit 1;

type="hq"  # for MUSDB18-HQ
subset="all"

musdb18_root="${data_root}/MUSDB18-HQ"

audyn-download-musdb18 \
type="${type}" \
root="${data_root}" \
musdb18_root="${musdb18_root}"

audyn-decode-musdb18 \
mp4_root="${musdb18_root}" \
wav_root="${musdb18_root}" \
subset="${subset}"
