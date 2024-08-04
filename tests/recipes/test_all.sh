#!/bin/bash

set -eu
set -o pipefail

(
    cd test_parse_run_command/
    echo "start: test_parse_run_command"
    . ./train.sh
    echo "end: test_parse_run_command"
    cd -
)
(
    cd test_decode_musdb18/
    echo "start: test_decode_musdb18"
    . ./preprocess.sh
    echo "end: test_decode_musdb18"
    cd -
)
