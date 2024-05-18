#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""

exp_root="./exp"
tensorboard_root="./tensorboard"

train_name="full_train"
validation_name="full_validation"

audioset_label_csv_url="http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
audioset_ontology_url="https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
audioset_balanced_train_csv_url="http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
audioset_unbalanced_train_csv_url="http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"
audioset_eval_csv_url="http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
data_root="../data"
dump_root="dump"

dump_format="webdataset"

system="defaults"
preprocess="audioset"
data="ssast"
train="ssast"
model="multitask_ssast_patch_mask400"
optimizer="adam"
lr_scheduler="ssast"
criterion="ssast"

. ../../_common/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1: Download dataset"

    (
        . ../_common/download.sh \
        --audioset-label-csv-url "${audioset_label_csv_url}" \
        --audioset-ontology-url "${audioset_ontology_url}" \
        --audioset-balanced-train-csv-url "${audioset_balanced_train_csv_url}" \
        --audioset-unbalanced-train-csv-url "${audioset_unbalanced_train_csv_url}" \
        --audioset-eval-csv-url "${audioset_eval_csv_url}" \
        --data-root "${data_root}"
    )

    (
        . ../_common/download_audio.sh \
        --audioset-balanced-train-csv-url "${audioset_balanced_train_csv_url}" \
        --audioset-unbalanced-train-csv-url "${audioset_unbalanced_train_csv_url}" \
        --audioset-eval-csv-url "${audioset_eval_csv_url}" \
        --data-root "${data_root}" \
        --preprocess "${preprocess}"
    )
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Preprocess dataset"

    (
        . ./preprocess.sh \
        --stage 0 \
        --stop-stage 2 \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --preprocess "${preprocess}" \
        --data "${data}"
    )
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Train SSAST"

    (
        . ./train.sh \
        --tag "${tag}" \
        --continue-from "${continue_from}" \
        --exp-root "${exp_root}" \
        --tensorboard-root "${tensorboard_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --train-name "${train_name}" \
        --validation-name "${validation_name}" \
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
