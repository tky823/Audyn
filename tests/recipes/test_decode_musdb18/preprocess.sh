#!/bin/bash

set -eu
set -o pipefail

is_ffmpeg_available="$(python local/is_ffmpeg_available.py)"

if [ "${is_ffmpeg_available}" = "False" ]; then
    echo "FFmpeg is not available."
    exit 0;
fi

musdb18_7s_url="https://zenodo.org/api/files/1ff52183-071a-4a59-923f-7a31c4762d43/MUSDB18-7-STEMS.zip"
data_root="./data"

type="7s"
musdb18_root="${data_root}/MUSDB18-7s"

audyn-download-musdb18 \
type="${type}" \
root="${data_root}" \
musdb18_root="${musdb18_root}"

audyn-decode-musdb18 \
mp4_root="${musdb18_root}" \
wav_root="${musdb18_root}" \
subset="validation"
