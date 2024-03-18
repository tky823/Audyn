#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""
checkpoint=""
text_to_feat_checkpoint=""
feat_to_wave_checkpoint=""

exp_root="./exp"
tensorboard_root="./tensorboard"

ljspeech_url="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
data_root="../data"
dump_root="dump"

dump_format="torch"

system="defaults"
preprocess="ljspeech_text-to-feat"
data="soundstream"
train="soundstream"
test="soundstream_reconstruction"
model="soundstream"
optimizer="soundstream"
lr_scheduler="soundstream"
criterion="soundstream"

n_validation=500
n_test=500

. ../../_common/parse_options.sh || exit 1;

ljspeech_root="${data_root}/LJSpeech-1.1"

set +u

# path to local scripts
export PYTHONPATH="./:${PYTHONPATH}"

set -u

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1"

    (
        . ../_common/download.sh \
        --ljspeech-url "${ljspeech_url}" \
        --data-root "${data_root}"
    )
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Preprocessing"

    (
        . ./preprocess.sh \
        --stage 0 \
        --stop-stage 4 \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --n-validation ${n_validation} \
        --n-test ${n_test}
    )
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Training"

    (
        . ./train.sh \
        --tag "${tag}" \
        --continue-from "${continue_from}" \
        --exp-root "${exp_root}" \
        --tensorboard-root "${tensorboard_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --train "${train}" \
        --model "${model}" \
        --optimizer "${optimizer}" \
        --lr-scheduler "${lr_scheduler}" \
        --criterion "${criterion}"
    )
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Reconstruction of signals"

    (
        . ./test_reconstruction.sh \
        --tag "${tag}" \
        --checkpoint "${checkpoint}" \
        --exp-root "${exp_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --test "${test}" \
        --model "${model}"
    )
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Save quantized features."

    (
        . ./save_quantized_features.sh \
        --tag "${tag}" \
        --checkpoint "${checkpoint}" \
        --exp-root "${exp_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --train "${train}" \
        --model "${model}"
    )
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Convert SoundStream to SoundStreamFirstStageDecoder"

    (
        . ./convert_soundstream.sh \
        --tag "${tag}" \
        --checkpoint "${checkpoint}" \
        --exp-root "${exp_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --train "${train}"
    )
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 5: Training SoundStream-TTS"

    (
        . ./train_tts.sh \
        --tag "${tag}" \
        --continue-from "${continue_from}" \
        --feat-to-wave-checkpoint "${feat_to_wave_checkpoint}" \
        --exp-root "${exp_root}" \
        --tensorboard-root "${tensorboard_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --train "${train}" \
        --model "${model}" \
        --optimizer "${optimizer}" \
        --lr-scheduler "${lr_scheduler}" \
        --criterion "${criterion}"
    )
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Stage 6: Synthesize speeches"

    (
        . ./test_tts.sh \
        --tag "${tag}" \
        --text-to-feat-checkpoint "${text_to_feat_checkpoint}" \
        --feat-to-wave-checkpoint "${feat_to_wave_checkpoint}" \
        --exp-root "${exp_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --system "${system}" \
        --preprocess "${preprocess}" \
        --data "${data}" \
        --test "${test}" \
        --model "${model}"
    )
fi
