#!/bin/bash

set -eu
set -o pipefail

musdb18_url="https://zenodo.org/records/1117372/files/musdb18.zip"
data_root="../data"

. ../../_common/parse_options.sh || exit 1;

musdb18_filename=$(basename "${musdb18_url}")
musdb18_dirname="${musdb18_filename/.zip/}"
musdb18_root="${data_root}/MUSDB18"

subset_names=(train test)

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

# from .mp4 to .wav
# ported from https://github.com/sigsep/sigsep-mus-io/blob/023615154797df0c7b40005f773fc9d977c19915/scripts/decode.sh.
cd ${musdb18_root}

for subset_name in "${subset_names[@]}" ; do
    cd ${subset_name}

    for stem in *.stem.mp4 ; do
        name="$(echo $stem | awk -F".stem.mp4" '{$0=$1}1')";

        if [ -e "${name}/mixture.wav" ] && [ -e "${name}/drums.wav" ] && [ -e "${name}/bass.wav" ] && [ -e "${name}/other.wav" ] && [ -e "${name}/vocals.wav" ]; then
            continue
        fi

        mkdir -p "${name}"
        cd "${name}"

        ffmpeg -loglevel panic -i "../${stem}" -map 0:0 -vn mixture.wav
        ffmpeg -loglevel panic -i "../${stem}" -map 0:1 -vn drums.wav
        ffmpeg -loglevel panic -i "../${stem}" -map 0:2 -vn bass.wav
        ffmpeg -loglevel panic -i "../${stem}" -map 0:3 -vn other.wav
        ffmpeg -loglevel panic -i "../${stem}" -map 0:4 -vn vocals.wav

        cd ../
    done

    cd ../
done
