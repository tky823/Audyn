#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_dir="./log"

preprocess="ljspeech_text-to-feat"
data="ljspeech_text-to-feat"

n_validation=5
n_test=5

. ../../_common/parse_options.sh || exit 1;

ljspeech_root="${data_root}/LJSpeech-1.1"
csv_path="${ljspeech_root}/metadata.csv"
wav_dir="${ljspeech_root}/wavs"

pair_root="${data_root}/pair"
textgrid_dir="${data_root}/textgrid"
dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"
symbols_path="${dump_dir}/symbols.pth"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Preprocess stage 0: Prepare for MFA."
    echo "To download G2P model, run 'mfa model download g2p english_us_arpa'."
    echo "To download dictionary, run 'mfa model download dictionary english_us_arpa'."
    echo "To download acoustic model, run 'mfa model download acoustic english_us_arpa'."
    echo "Then, set '--stage 1' and re-run."

    exit 0
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Copy wav & text files to ${pair_root}."

    mkdir -p "${pair_root}"

    IFS="|"
    while read filename text normalized_text; do
        cp "${wav_dir}/${filename}.wav" "${pair_root}/${filename}.wav"
        echo "${normalized_text}" > "${pair_root}/${filename}.txt"
    done < "${csv_path}"
    unset IFS
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Create textGrid files by MFA."

    corpus="english_us_arpa"

    mfa align "${pair_root}" "${corpus}" "${corpus}" "${textgrid_dir}" \
    --clean
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Preprocess stage 3: Split data into training/validation/test"

    tmp_root="/tmp/$(uuidgen)"

    mkdir -p "${tmp_root}"

    n_all=20
    n_train=$((${n_all} - ${n_validation} - ${n_test}))
    head -n ${n_train} "${csv_path}" > "${tmp_root}/train.csv"
    set +e
    tail -n $((${n_validation} + ${n_test})) "${csv_path}" | head -n ${n_validation} > "${tmp_root}/validation.csv"
    set -e
    tail -n ${n_test} "${csv_path}" > "${tmp_root}/test.csv"

    mkdir -p "${list_dir}"

    IFS="|"

    for subset in train validation test; do
        : > "${list_dir}/${subset}.txt"
        while read filename text normalized_text; do
            echo "${filename}" >> "${list_dir}/${subset}.txt"
        done < "${tmp_root}/${subset}.csv"
    done

    unset IFS

    rm -r "${tmp_root}"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Preprocess stage 4: Save features"

    python ./local/save_symbols.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.symbols_path="${symbols_path}"

    for subset in train validation test; do
        python ./local/save_features.py \
        --config-dir "./conf" \
        hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
        preprocess="${preprocess}" \
        data="${data}" \
        preprocess.list_path="${list_dir}/${subset}.txt" \
        preprocess.wav_dir="${wav_dir}" \
        preprocess.textgrid_dir="${textgrid_dir}" \
        preprocess.feature_dir="${feature_dir}" \
        preprocess.symbols_path="${symbols_path}"
    done
fi
